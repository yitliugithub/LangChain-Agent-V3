import os
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch
from langchain_core.messages import AIMessageChunk


# ============================================================
# 1. 页面设置
# ============================================================

st.set_page_config(
    page_title="小明 AI Agent",
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


# 检查 API Key
if not DEEPSEEK_API_KEY:
    st.error("❌ 缺少 DEEPSEEK_API_KEY，请检查 .env 文件。")
    st.stop()

if not DEEPSEEK_BASE_URL:
    st.error("❌ 缺少 DEEPSEEK_BASE_URL，请检查 .env 文件。")
    st.stop()

if not TAVILY_API_KEY:
    st.error("❌ 缺少 TAVILY_API_KEY，请检查 .env 文件。")
    st.stop()


# ============================================================
# 3. 创建 DeepSeek Model
# ============================================================

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
)


# ============================================================
# 4. 定义 Tools
# ============================================================

@tool
def calculator(operation: str, a: float, b: float) -> float:
    """
    执行两个数字之间的基础数学运算。

    operation 支持：

    - add
    - subtract
    - multiply
    - divide
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


@tool
def get_current_time() -> str:
    """
    获取当前计算机本地的日期和时间。
    """

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Tavily Search Tool
web_search = TavilySearch(
    max_results=5
)


# Tool List
tools = [
    calculator,
    get_current_time,
    web_search
]


# ============================================================
# 5. System Prompt
# ============================================================

SYSTEM_PROMPT = """
你叫小明，是一个友好、可靠、专业的 AI 学习助手。

你的主要职责是帮助用户学习知识、理解概念、
解决问题以及完成合理的学习任务。


【工具使用规则】

1. 如果用户要求进行数学计算，
   优先使用 calculator 工具。

2. 如果用户询问当前日期、当前时间，
   使用 get_current_time 工具。

3. 如果用户询问：

   - 最新新闻
   - 实时信息
   - 当前事件
   - 最新技术发展
   - 可能已经发生变化的信息

   应使用 web_search 工具搜索最新信息。

4. 如果问题属于普通知识，
   并且不需要外部实时信息，
   直接使用你的知识回答。

5. 不要为了展示工具能力而进行没有必要的工具调用。

6. 不允许编造 Tool 的执行结果。

7. 如果 Tool 调用失败，
   应明确告诉用户发生了错误。


【回答要求】

- 默认使用中文回答。
- 技术术语可以保留英文。
- 用户是初学者时，应尽量解释清楚。
- 复杂知识应合理拆解。
- 涉及代码时，先解释基本思路，再展示代码。
- 不确定的信息不要假装确定。
"""


# ============================================================
# 6. 初始化 Session State
# ============================================================

# UI 聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []


# LangGraph Memory
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()


# Conversation Thread ID
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# ============================================================
# 7. 创建 Agent
# ============================================================

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=st.session_state.checkpointer
)


# Agent Config
config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# ============================================================
# 8. Sidebar
# ============================================================

with st.sidebar:

    st.title("🤖 小明 AI Agent")

    st.caption(
        "LangChain + DeepSeek Agent"
    )

    st.divider()

    st.subheader("当前功能")

    st.write("✓ AI 对话")
    st.write("✓ 数学计算")
    st.write("✓ 当前时间")
    st.write("✓ 实时 Web Search")
    st.write("✓ Short-term Memory")
    st.write("✓ Streaming")

    st.divider()

    # ========================================================
    # New Chat
    # ========================================================

    if st.button(
        "＋ 新建对话",
        use_container_width=True
    ):

        # 删除旧 thread 的 checkpoint
        old_thread_id = st.session_state.thread_id

        try:
            st.session_state.checkpointer.delete_thread(
                old_thread_id
            )
        except Exception:
            pass

        # 清空网页聊天记录
        st.session_state.messages = []

        # 创建新的 conversation ID
        st.session_state.thread_id = str(
            uuid.uuid4()
        )

        # Streamlit 重新运行
        st.rerun()

    st.divider()

    st.caption("Conversation ID")

    st.code(
        st.session_state.thread_id[:8] + "..."
    )


# ============================================================
# 9. 主页面
# ============================================================

st.title("🤖 小明 AI Agent")

st.caption(
    "基于 LangChain + DeepSeek 构建的智能学习助手"
)


# ============================================================
# 10. 初始欢迎界面
# ============================================================

if len(st.session_state.messages) == 0:

    st.info(
        """
        你好，我是小明。

        你可以尝试问我：

        - 什么是 LangChain Agent？
        - 帮我计算 125.5 × 32.4
        - 现在几点？
        - 今天有什么最新的 AI 新闻？
        """
    )


# ============================================================
# 11. 显示历史聊天
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 12. 用户输入
# ============================================================

user_input = st.chat_input(
    "请输入你的问题..."
)


# ============================================================
# 13. Agent Streaming Response
# ============================================================

if user_input:

    # --------------------------------------------------------
    # 保存 User Message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # --------------------------------------------------------
    # 显示 User Message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_input
        )


    # --------------------------------------------------------
    # Assistant Message
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        try:

            # 最终回答
            reply = ""

            response_placeholder = st.empty()

            tool_status_placeholder = st.empty()

            tool_labels = {

                "calculator":
                    "🧮 正在进行数学计算...",

                "get_current_time":
                    "🕐 正在获取当前时间...",

                "tavily_search":
                    "🔎 正在搜索互联网...",

                "web_search":
                    "🔎 正在搜索互联网..."
            }

            for message_chunk, metadata in agent.stream(

                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_input
                            }
                        ]
                    },

                    config=config,

                    stream_mode="messages"
            ):

                # =============================================
                # 1. 检测 Tool Call
                # =============================================

                tool_calls = getattr(
                    message_chunk,
                    "tool_calls",
                    None
                )

                if tool_calls:

                    for tool_call in tool_calls:
                        tool_name = tool_call.get(
                            "name",
                            "unknown_tool"
                        )

                        status_text = tool_labels.get(
                            tool_name,
                            f"⚙️ 正在调用工具：{tool_name}..."
                        )

                        tool_status_placeholder.info(
                            status_text
                        )

                # =============================================
                # 2. 只显示 AIMessageChunk
                # =============================================

                if isinstance(message_chunk, AIMessageChunk):

                    content = message_chunk.content

                    # 只处理普通字符串文本
                    if isinstance(content, str) and content:
                        reply += content

                        response_placeholder.markdown(
                            reply + "▌"
                        )

            # =============================================
            # Agent 完成
            # =============================================

            tool_status_placeholder.empty()

            response_placeholder.markdown(reply)


        except Exception as e:

            reply = (
                f"Agent 调用失败：{e}"
            )

            st.error(
                reply
            )


    # ========================================================
    # 14. 保存 Assistant Message
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply
        }
    )