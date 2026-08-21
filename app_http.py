# ============================================================
# V4 / HTTP VERSION
# 与 V3 最大区别：
# 当前版本直接通过 HTTP 请求 DeepSeek API，
# 1. Conversation Memory
# 2. Tool Calling
# 3. Tool Execution
# 4. Tool Result 回传
# 5. Agent Loop
# 6. Streaming
#
# ============================================================


import os
import json
import uuid
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv
from search_rag import search_rag
from sql_agent import (
    generate_sql,
    validate_sql,
    query_mysql
)

# ============================================================
# 1. Streamlit 页面设置
# ============================================================

st.set_page_config(
    page_title="小明 AI Agent - HTTP Version",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# 2. 加载环境变量
# ============================================================

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


if not DEEPSEEK_API_KEY:
    st.error("缺少 DEEPSEEK_API_KEY，请检查 .env")
    st.stop()

if not DEEPSEEK_BASE_URL:
    st.error("缺少 DEEPSEEK_BASE_URL，请检查 .env")
    st.stop()

if not TAVILY_API_KEY:
    st.error("缺少 TAVILY_API_KEY，请检查 .env")
    st.stop()


# ============================================================
# 3. DeepSeek API Endpoint
# ============================================================

DEEPSEEK_CHAT_URL = (
    DEEPSEEK_BASE_URL.rstrip("/")
    + "/chat/completions"
)
MODEL_NAME = "deepseek-v4-flash"


# ============================================================
# 4. System Prompt
# ============================================================


SYSTEM_PROMPT = """
你叫小明，是一个友好、可靠、专业的 AI 学习助手。

你的主要职责是帮助用户学习知识、理解概念、
解决问题以及完成合理的学习任务。

【工具使用规则】

工具使用规则：

1. 数学计算问题使用 calculator。
2. 当前时间问题使用 get_current_time。
3. 最新新闻、实时事件、互联网动态使用 web_search。
4. 当用户询问内部研究报告、品牌营销趋势、
   消费者洞察、营销观点等非结构化知识时，
   使用 search_knowledge_base。
5. 当用户询问数据库中的结构化数据，例如：
   - 品牌
   - 帖子
   - 达人
   - 点赞数
   - 评论数
   - 收藏数
   - 浏览量
   - 排名
   - 平均值
   - Campaign
   - ROI
   - 投放表现
   使用 query_marketing_database。
6. 如果问题需要研究报告内容，优先使用知识库；
   如果问题需要精确数字或数据库统计，优先使用 MySQL。

7. query_marketing_database 返回 SQL 和真实数据库查询结果。
   最终回答只能根据这些返回结果回答，
   不得修改、补充或编造数据库中的数字。

8. search_knowledge_base 返回知识库检索片段。
   最终回答需要基于这些片段进行总结。
   如果知识库证据不足，需要明确说明。

9. 不需要工具的问题直接回答。

10. 不允许编造工具执行结果。

11. 工具失败时明确告诉用户。

【回答要求】

- 默认使用中文。
- 技术术语可以保留英文。
- 面向初学者时解释清楚。
- 涉及代码时先解释思路。
- 不确定的信息不要假装确定。
"""


# ============================================================
# 5. 定义 Python Tools
# ============================================================

def calculator(operation: str, a: float, b: float):
    """
    执行基础数学运算。
    """

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":

        if b == 0:
            raise ValueError("除数不能为 0")

        return a / b

    else:
        raise ValueError(
            "operation 必须是 add、subtract、multiply 或 divide"
        )


def get_current_time():
    """
    获取当前计算机本地时间。
    """
    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# 6. Tavily Web Search
# ============================================================

def web_search(query: str):
    """
    直接通过 Tavily HTTP API 搜索互联网。
    """

    url = "https://api.tavily.com/search"

    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "query": query,
        "search_depth": "basic",
        "max_results": 5
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    # --------------------------------------------
    # 只提取真正有用的信息：
    #
    # title
    # url
    # content
    # --------------------------------------------

    simplified_results = []

    for item in data.get("results", []):

        simplified_results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")
            }
        )

    return simplified_results

def search_knowledge_base(query: str):
    """
    搜索本地 RAG 知识库，返回最相关的文档片段。
    """

    results = search_rag(
        question=query,
        top_k=3
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    rag_results = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        rag_results.append(
            {
                "source": metadata.get(
                    "source",
                    "Unknown"
                ),
                "section": metadata.get(
                    "section",
                    "Unknown"
                ),
                "content": document,
                "distance": float(distance)
            }
        )

    return rag_results
def query_marketing_database(question: str):
    """
    将自然语言问题转换为只读 SQL，
    查询 MySQL，
    返回真实数据库结果给外层 Agent。
    """

    # 1. LLM 生成 SQL
    sql = generate_sql(
        question
    )

    # 3. 安全检查
    validate_sql(
        sql
    )

    # 4. 查询数据库
    rows = query_mysql(
        sql
    )

    # 5. 不在这里生成自然语言答案
    #    直接返回原始查询结果
    return {
        "question": question,
        "sql": sql,
        "row_count": len(rows),
        "rows": rows
    }
# ============================================================
# 7. Tool Schema
# ============================================================
# 因为没有 LangChain，
# 我们必须自己把 Tool 的：
#
# - name
# - description
# - parameters
#
# 写成 JSON Schema。
#
# 然后直接发送给 DeepSeek API。
#
# ============================================================


TOOLS = [

    # --------------------------------------------------------
    # calculator
    # --------------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "calculator",

            "description":
                "执行两个数字之间的基础数学运算。",

            "parameters": {

                "type": "object",

                "properties": {

                    "operation": {
                        "type": "string",
                        "enum": [
                            "add",
                            "subtract",
                            "multiply",
                            "divide"
                        ],
                        "description":
                            "需要执行的数学运算"
                    },

                    "a": {
                        "type": "number",
                        "description": "第一个数字"
                    },

                    "b": {
                        "type": "number",
                        "description": "第二个数字"
                    }
                },

                "required": [
                    "operation",
                    "a",
                    "b"
                ]
            }
        }
    },


    # --------------------------------------------------------
    # get_current_time
    # --------------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "get_current_time",

            "description":
                "获取当前计算机本地的日期和时间。",

            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },


    # --------------------------------------------------------
    # web_search
    # --------------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "web_search",

            "description":
                "搜索互联网获取最新新闻、实时信息、"
                "当前事件以及可能已经发生变化的信息。",

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {
                        "type": "string",
                        "description":
                            "需要在互联网搜索的查询内容"
                    }

                },

                "required": [
                    "query"
                ]
            }
        }
    },
    # --------------------------------------------------------
    # search_knowledge_base
    # --------------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "search_knowledge_base",

            "description":
                "搜索内部营销研究 RAG 知识库。"
                "当用户询问品牌营销趋势、营销洞察、"
                "广告主行为、营销投资、消费者趋势等"
                "知识库中可能存在的信息时使用。",

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {
                        "type": "string",
                        "description":
                            "需要在内部知识库中检索的问题"
                    }

                },

                "required": [
                    "query"
                ]
            }
        }
    },
    {
        "type": "function",

        "function": {

            "name": "query_marketing_database",

            "description":
                "查询内部 MySQL 营销数据库并返回真实 SQL 查询结果。"
                "当用户询问品牌、帖子、达人、互动量、"
                "浏览量、Campaign、投放表现、ROI、"
                "排名、平均值、数量统计等结构化数据时使用。"
                "该工具返回 SQL、结果行数和数据库原始结果，"
                "请基于返回结果生成最终回答。",

            "parameters": {

                "type": "object",

                "properties": {

                    "question": {
                        "type": "string",
                        "description":
                            "需要使用内部 MySQL 数据库回答的问题"
                    }

                },

                "required": [
                    "question"
                ]
            }
        }
    }

]


# ============================================================
# 8. Tool Router
# ============================================================
# 当前版本：
#
# 我们必须自己写 Router。
#
# ============================================================


def execute_tool(tool_name: str, arguments: dict):
    """
    根据模型请求的工具名称执行对应 Python Function。
    """

    if tool_name == "calculator":

        return calculator(
            operation=arguments["operation"],
            a=arguments["a"],
            b=arguments["b"]
        )


    elif tool_name == "get_current_time":

        return get_current_time()


    elif tool_name == "web_search":

        return web_search(
            query=arguments["query"]
        )

    elif tool_name == "search_knowledge_base":

        return search_knowledge_base(
            query=arguments["query"]
        )
    elif tool_name == "query_marketing_database":

        result = query_marketing_database(
            question=arguments["question"]
        )

        return result
    else:

        raise ValueError(
            f"未知工具：{tool_name}"
        )


# ============================================================
# 9. DeepSeek HTTP 请求函数
# ============================================================
# 当前：
#
# requests.post(...)
#
#
# 这里是真正的 HTTP 调用。
#
# ============================================================


def call_deepseek(messages, tools=None, stream=False):
    """
    直接调用 DeepSeek Chat Completions HTTP API。
    """

    headers = {

        "Authorization":
            f"Bearer {DEEPSEEK_API_KEY}",

        "Content-Type":
            "application/json"
    }


    payload = {

        "model":
            MODEL_NAME,

        "messages":
            messages,

        "stream":
            stream
    }


    # 如果允许 Tool Calling
    if tools is not None:

        payload["tools"] = tools

        # 让模型自己决定：
        #
        # 直接回答
        # 或
        # 调用 Tool
        payload["tool_choice"] = "auto"


    response = requests.post(

        DEEPSEEK_CHAT_URL,

        headers=headers,

        json=payload,

        stream=stream,

        timeout=120
    )


    response.raise_for_status()

    return response


# ============================================================
# 10. 手写 Agent Loop
# ============================================================
# 【当前版本】
#
# User
#   ↓
# HTTP → LLM
#   ↓
# LLM 是否有 tool_calls？
#
# NO
#   ↓
# Final Answer
#
# YES
#   ↓
# Python Tool
#   ↓
# Tool Result
#   ↓
# messages.append(tool result)
#   ↓
# 再次 HTTP → LLM
#   ↓
# Loop
#
# ============================================================


def run_agent(messages, tool_status_placeholder):
    """
    手动运行 Agent。

    返回更新后的 messages。
    """


    # 防止模型无限 Tool Loop
    MAX_AGENT_STEPS = 6


    for step in range(MAX_AGENT_STEPS):


        # ====================================================
        # STEP 1
        #
        # 请求 DeepSeek
        # ====================================================

        response = call_deepseek(

            messages=messages,

            tools=TOOLS,

            stream=False
        )


        data = response.json()


        assistant_message = (
            data["choices"][0]["message"]
        )


        # ====================================================
        # STEP 2
        #
        # 检查模型是否要求调用 Tool
        # ====================================================

        tool_calls = assistant_message.get(
            "tool_calls"
        )


        # ====================================================
        # CASE A：
        #
        # 没有 Tool Call
        #
        # 说明模型已经准备直接回答。
        # ====================================================

        if not tool_calls:

            return assistant_message


        # ====================================================
        # CASE B：
        #
        # 有 Tool Calls
        # ====================================================
        # --------------------------------------------
        #
        # 必须先把模型这个带 tool_calls 的
        # assistant message 放回 conversation history。
        #
        # V3 里 LangChain 自动做。
        # 现在我们必须自己做。
        # --------------------------------------------

        messages.append(
            assistant_message
        )


        # --------------------------------------------
        # 一个 assistant message
        # 可能要求多个 tool call。
        # --------------------------------------------

        for tool_call in tool_calls:


            tool_call_id = tool_call["id"]

            function_data = tool_call["function"]

            tool_name = function_data["name"]
            # ----------------------------------------
            # arguments 通常是 JSON String
            #
            # 例如：
            #
            # '{"a":125.5,"b":32.4,
            #   "operation":"multiply"}'
            #
            # 所以需要 json.loads()
            # ----------------------------------------

            raw_arguments = function_data.get(
                "arguments",
                "{}"
            )

            try:

                arguments = json.loads(
                    raw_arguments
                )

            except json.JSONDecodeError:

                arguments = {}

            # =================================================
            # 显示 Tool 状态
            # =================================================

            tool_labels = {

                "calculator":
                    "🧮 正在进行数学计算...",

                "get_current_time":
                    "🕐 正在获取当前时间...",

                "web_search":
                    "🔎 正在搜索互联网...",

                "search_knowledge_base":
                    "📚 正在搜索内部知识库...",

                "query_marketing_database":
                    "🗄️ 正在查询内部营销数据库..."
            }


            tool_status_placeholder.info(

                tool_labels.get(

                    tool_name,

                    f"⚙️ 正在调用 {tool_name}..."
                )
            )

            # =================================================
            # STEP 3
            #
            # 真正执行 Python Tool
            # =================================================

            try:

                tool_result = execute_tool(

                    tool_name,

                    arguments
                )


            except Exception as e:

                tool_result = {
                    "error": str(e)
                }


            # =================================================
            # STEP 4
            #
            # 把 Tool Result 加回 Messages
            # =================================================
            #
            # 【现在】
            #
            # 我们自己构造：
            #
            # role = tool
            # tool_call_id = ...
            # content = ...
            #
            # =================================================

            messages.append(

                {
                    "role": "tool",

                    "tool_call_id":
                        tool_call_id,

                    "content":
                        json.dumps(
                            tool_result,
                            ensure_ascii=False
                        )
                }
            )


        # --------------------------------------------
        # for-loop 回到顶部
        #
        # 重新 HTTP 请求 DeepSeek。
        #
        # 此时模型会看到：
        #
        # HumanMessage
        # Assistant Tool Call
        # Tool Result
        #
        # 然后根据 Tool Result 再作决定。
        # --------------------------------------------


    # ========================================================
    # 如果超过 MAX_AGENT_STEPS
    # ========================================================

    return {
        "role": "assistant",
        "content":
            "Agent 执行步骤过多，为避免无限循环已停止。"
    }


# ============================================================
# 11. Streaming Final Answer
# ============================================================
#
# 前面的 Tool Loop 暂时使用普通 HTTP 请求，
# 因为我们需要先明确得到完整 tool_calls。
#
# 最终回答再使用 stream=True。
#
# ============================================================


def stream_final_answer(messages):
    """
    通过 DeepSeek HTTP API 流式生成最终回答。
    """


    response = call_deepseek(

        messages=messages,

        # 最终回答阶段不再提供 tools
        tools=None,

        stream=True
    )


    for line in response.iter_lines(
        decode_unicode=True
    ):


        if not line:
            continue


        if not line.startswith("data:"):
            continue


        data_string = line[len("data:"):].strip()


        if data_string == "[DONE]":
            break


        try:

            chunk = json.loads(
                data_string
            )

        except json.JSONDecodeError:
            continue


        choices = chunk.get(
            "choices",
            []
        )


        if not choices:
            continue


        delta = choices[0].get(
            "delta",
            {}
        )


        content = delta.get(
            "content"
        )


        if content:

            yield content


# ============================================================
# 12. Session State
# ============================================================
# 当前：
#
# 我们自己把 messages 放进：
#
# st.session_state
#
# Memory 的本质就是：
#
# 每次 API 请求时重新发送历史 messages。
#
# DeepSeek API 本身不会替我们保存会话。
#
# ============================================================


if "messages" not in st.session_state:

    st.session_state.messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }

    ]


if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(
        uuid.uuid4()
    )


# ============================================================
# 13. Sidebar
# ============================================================

with st.sidebar:

    st.title("🤖 小明")

    st.caption(
        "Direct HTTP API Version"
    )

    st.divider()


    st.subheader("Architecture")

    st.write("✓ Streamlit")
    st.write("✓ Direct HTTP API")
    st.write("✓ DeepSeek")
    st.write("✓ Manual Agent Loop")
    st.write("✓ Manual Tool Calling")
    st.write("✓ Tavily HTTP API")
    st.write("✓ Manual Memory")
    st.write("✓ Streaming")


    st.divider()


    # ========================================================
    # New Chat
    # ========================================================

    if st.button(
        "＋ 新建对话",
        use_container_width=True
    ):

        # --------------------------------------------
        # 清除历史对话
        #
        # 但 System Prompt 必须重新保留。
        # --------------------------------------------

        st.session_state.messages = [

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }

        ]


        st.session_state.thread_id = str(
            uuid.uuid4()
        )


        st.rerun()


    st.divider()

    st.caption("Conversation ID")

    st.code(
        st.session_state.thread_id[:8]
        + "..."
    )


# ============================================================
# 14. 页面标题
# ============================================================

st.title(
    "🤖 小明 AI Agent"
)

st.caption(
    "Direct HTTP API · No LangChain"
)


# ============================================================
# 15. Welcome
# ============================================================

# 减去 System Message
visible_messages = [

    message

    for message
    in st.session_state.messages

    if message.get("role")
    in ["user", "assistant"]
]


if len(visible_messages) == 0:

    st.info(
        """
        你好，我是小明。

        当前版本不使用 LangChain，
        而是直接通过 HTTP API 调用 DeepSeek。

        你可以尝试：

        - 什么是 Agent？
        - 计算 125.5 × 32.4
        - 现在几点？
        - 今天有什么最新 AI 新闻？
        """
    )


# ============================================================
# 16. 显示 Conversation History
# ============================================================
#
# Tool Messages 不展示在 UI。
#
# 这里只显示：
#
# user
# assistant
#
# ============================================================

for message in st.session_state.messages:

    role = message.get("role")


    if role not in [
        "user",
        "assistant"
    ]:
        continue


    # 带 tool_calls 的 assistant 中间消息
    # 也不直接显示给用户。

    if role == "assistant" and message.get(
        "tool_calls"
    ):

        continue


    content = message.get(
        "content"
    )


    if not content:
        continue


    with st.chat_message(role):

        st.markdown(content)


# ============================================================
# 17. Chat Input
# ============================================================

user_input = st.chat_input(
    "请输入你的问题..."
)


# ============================================================
# 18. 用户发送消息
# ============================================================

if user_input:


    # --------------------------------------------------------
    # 保存用户消息
    # --------------------------------------------------------

    user_message = {

        "role":
            "user",

        "content":
            user_input
    }


    st.session_state.messages.append(
        user_message
    )


    # --------------------------------------------------------
    # 显示用户消息
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_input
        )


    # ========================================================
    # 19. Assistant
    # ========================================================

    with st.chat_message("assistant"):


        tool_status_placeholder = st.empty()

        response_placeholder = st.empty()


        try:


            # =================================================
            # STEP 1
            #
            # 运行手写 Agent Loop
            # =================================================
            #
            # 注意：
            #
            # 这里传进去的是实际 history list。
            #
            # 如果模型调用 Tool：
            #
            # run_agent 会自动添加：
            #
            # assistant tool call
            # tool result
            #
            # =================================================

            final_message = run_agent(

                st.session_state.messages,

                tool_status_placeholder
            )


            # =================================================
            # STEP 2
            #
            # 如果 Agent 已经得到普通 Final Answer
            # =================================================

            final_content = final_message.get(
                "content"
            )


            # 清理 Tool Status
            tool_status_placeholder.empty()


            # -------------------------------------------------
            # 这里为了让最终 UI 保持 Streaming：
            #
            # 如果前面的 run_agent 已经直接返回普通文本，
            # 我们可以直接显示。
            #
            # 如果经历了 Tool Calling，
            # 也可以进一步让模型进行一次最终流式整理。
            # -------------------------------------------------


            if final_content:


                reply = ""

                # ---------------------------------------------
                # 这里直接模拟逐步显示是不必要的。
                #
                # 最基础版本直接显示最终内容。
                #
                # 普通知识问答也会正常工作。
                # ---------------------------------------------

                reply = final_content

                response_placeholder.markdown(
                    reply
                )


            else:

                reply = (
                    "模型没有返回可显示的回答。"
                )

                response_placeholder.markdown(
                    reply
                )


            # =================================================
            # STEP 3
            #
            # 把 Final Assistant Answer 加入 Memory
            # =================================================

            st.session_state.messages.append(

                {
                    "role":
                        "assistant",

                    "content":
                        reply
                }

            )


        except requests.HTTPError as e:


            reply = (
                "HTTP 请求失败："
                + str(e)
            )

            st.error(reply)


        except Exception as e:


            reply = (
                "Agent 执行失败："
                + str(e)
            )

            st.error(reply)