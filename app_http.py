import json
import os
from datetime import datetime
from decimal import Decimal

import requests
from dotenv import load_dotenv

from search_rag import search_rag
from sql_agent import generate_sql, query_mysql, validate_sql


load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

MODEL_NAME = "deepseek-v4-flash"
MAX_AGENT_STEPS = 6
DEEPSEEK_CHAT_URL = (
    DEEPSEEK_BASE_URL.rstrip("/") + "/chat/completions"
    if DEEPSEEK_BASE_URL
    else ""
)


SYSTEM_PROMPT = """
你是 Research Insight Agent，一个面向营销研究和品牌投放分析的 AI 助手。

你可以根据用户问题决定是否调用工具。工具选择原则：

1. calculator:
   用于明确的数学计算。

2. get_current_time:
   用于当前日期、当前时间、本地时间问题。

3. web_search:
   用于最新、实时、新闻、互联网动态、今天发生的事情。

4. search_knowledge_base:
   用于内部报告、营销趋势、消费者洞察、营销研究、白皮书、历史案例等非结构化知识。
   该工具返回 retrieved evidence。最终回答必须基于 evidence 总结。
   如果证据不足，必须明确说明知识库证据不足。

5. query_marketing_database:
   用于品牌、帖子、达人、Campaign、投放表现、ROI、点赞、评论、收藏、分享、浏览量、
   排名、平均值、数量统计等结构化 MySQL 数据。
   该工具返回真实 database result。最终回答不能修改、补充或编造数据库数字。

回答要求：
- 默认使用中文。
- 不需要工具的问题可以直接回答。
- 不要为了展示工具能力而调用工具。
- 不允许编造工具执行结果。
- 工具失败时要明确告诉用户。
- 对初学者保持清晰、简洁、可解释。
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行两个数字之间的基础数学运算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "需要执行的数学运算。",
                    },
                    "a": {"type": "number", "description": "第一个数字。"},
                    "b": {"type": "number", "description": "第二个数字。"},
                },
                "required": ["operation", "a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前计算机本地日期和时间。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网获取最新新闻、实时信息、当前事件和可能变化的信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要在互联网搜索的查询内容。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "搜索内部营销研究 RAG 知识库。适用于品牌营销趋势、消费者洞察、"
                "广告主行为、营销投资、营销研究和内部报告问题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要在内部知识库中检索的问题。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_marketing_database",
            "description": (
                "查询内部 MySQL 营销数据库并返回真实 SQL 查询结果。适用于品牌、帖子、"
                "达人、互动量、浏览量、Campaign、投放表现、ROI、排名、平均值和数量统计。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "需要使用内部 MySQL 数据库回答的问题。",
                    }
                },
                "required": ["question"],
            },
        },
    },
]


def require_env() -> None:
    required_env = {
        "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
        "DEEPSEEK_BASE_URL": DEEPSEEK_BASE_URL,
        "TAVILY_API_KEY": TAVILY_API_KEY,
    }
    missing = [key for key, value in required_env.items() if not value]
    if missing:
        raise ValueError("缺少环境变量：" + ", ".join(missing))


def json_default(value):
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def calculator(operation: str, a: float, b: float):
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ValueError("除数不能为 0")
        return a / b
    raise ValueError("operation 必须是 add、subtract、multiply 或 divide")


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def web_search(query: str):
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "search_depth": "basic",
        "max_results": 5,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
        }
        for item in data.get("results", [])
    ]


def search_knowledge_base(query: str):
    results = search_rag(question=query, top_k=3)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    rag_results = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        rag_results.append(
            {
                "source": metadata.get("source", "Unknown"),
                "section": metadata.get("section", "Unknown"),
                "chunk_index": metadata.get("chunk_index"),
                "chunk_method": metadata.get("chunk_method"),
                "content": document,
                "distance": float(distance),
                "distance_note": (
                    "Chroma 返回的距离值只用于排序参考；数值越小通常表示越接近，"
                    "不要把它当作百分比置信度。"
                ),
            }
        )

    return rag_results


def query_marketing_database(question: str):
    sql = generate_sql(question)
    validate_sql(sql)
    rows = query_mysql(sql)
    return {
        "question": question,
        "sql": sql,
        "row_count": len(rows),
        "rows": rows,
    }


def execute_tool(tool_name: str, arguments: dict):
    if tool_name == "calculator":
        return calculator(
            operation=arguments["operation"],
            a=arguments["a"],
            b=arguments["b"],
        )
    if tool_name == "get_current_time":
        return get_current_time()
    if tool_name == "web_search":
        return web_search(query=arguments["query"])
    if tool_name == "search_knowledge_base":
        return search_knowledge_base(query=arguments["query"])
    if tool_name == "query_marketing_database":
        return query_marketing_database(question=arguments["question"])
    raise ValueError(f"未知工具：{tool_name}")


def call_deepseek(messages, tools=None):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
    }

    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    response = requests.post(
        DEEPSEEK_CHAT_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]


def parse_tool_arguments(raw_arguments: str) -> dict:
    if not raw_arguments:
        return {}
    try:
        return json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"工具参数不是合法 JSON：{raw_arguments}") from exc


def append_tool_result(messages, tool_call_id: str, tool_result) -> None:
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(
                tool_result,
                ensure_ascii=False,
                default=json_default,
            ),
        }
    )


def run_agent(messages):
    """
    手写 Agent Loop。

    run_agent 会把 assistant tool call、tool result 和最终 assistant answer
    都追加到同一个 messages 列表中，CLI 主循环不要重复 append assistant answer。
    """
    for _ in range(MAX_AGENT_STEPS):
        assistant_message = call_deepseek(messages=messages, tools=TOOLS)
        tool_calls = assistant_message.get("tool_calls")

        if not tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.get("content", ""),
                }
            )
            return assistant_message.get("content", "")

        messages.append(assistant_message)

        for tool_call in tool_calls:
            tool_call_id = tool_call["id"]
            function_data = tool_call["function"]
            tool_name = function_data["name"]
            raw_arguments = function_data.get("arguments", "{}")

            print(f"[Tool] 正在调用 {tool_name}")

            try:
                arguments = parse_tool_arguments(raw_arguments)
                tool_result = execute_tool(tool_name, arguments)
            except Exception as exc:
                tool_result = {"error": str(exc)}

            append_tool_result(messages, tool_call_id, tool_result)

    final_text = "Agent 执行步骤过多，为避免无限循环已停止。"
    messages.append({"role": "assistant", "content": final_text})
    return final_text


def new_conversation():
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def main():
    require_env()

    messages = new_conversation()
    print("Research Insight Agent 已启动。")
    print("输入 quit 退出；输入 new 清空当前 conversation memory。\n")

    while True:
        user_input = input("User: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("程序结束。")
            break
        if user_input.lower() == "new":
            messages = new_conversation()
            print("已清空当前 conversation memory。\n")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            reply = run_agent(messages)
        except requests.HTTPError as exc:
            reply = f"HTTP 请求失败：{exc}"
            messages.append({"role": "assistant", "content": reply})
        except Exception as exc:
            reply = f"Agent 执行失败：{exc}"
            messages.append({"role": "assistant", "content": reply})

        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
