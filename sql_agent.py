# Author: Yiting Liu
# Time 21/8/2026 PM5:50

# ============================================================
# Research Insight Agent - SQL Tool Module
#
# 功能：
#
# 用户自然语言问题
#        ↓
# DeepSeek 生成 SQL
#        ↓
# Python 检查 SQL 安全性
#        ↓
# MySQL 执行查询
#        ↓
# 返回真实数据库结果给外层 Agent
#
# 当前文件作为 app_http.py 的 SQL 工具模块使用。
# 最终自然语言回答统一由外层 Agent 生成。
#
# 当前版本完全不使用 LangChain。
# DeepSeek 通过 HTTP POST 直接调用。
# ============================================================


import os
import re

import requests
import mysql.connector
from dotenv import load_dotenv


# ============================================================
# 1. 加载环境变量
# ============================================================

load_dotenv(override=True)


# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")


# MySQL
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")


# ============================================================
# 2. 检查环境变量
# ============================================================

required_env = {
    "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
    "DEEPSEEK_BASE_URL": DEEPSEEK_BASE_URL,
    "MYSQL_HOST": MYSQL_HOST,
    "MYSQL_USER": MYSQL_USER,
    "MYSQL_PASSWORD": MYSQL_PASSWORD,
    "MYSQL_DATABASE": MYSQL_DATABASE,
}


for key, value in required_env.items():

    if not value:
        raise ValueError(
            f"缺少环境变量：{key}，请检查 .env 文件"
        )


# ============================================================
# 3. DeepSeek API 地址
# ============================================================

DEEPSEEK_CHAT_URL = (
    DEEPSEEK_BASE_URL.rstrip("/")
    + "/chat/completions"
)

MODEL_NAME = "deepseek-v4-flash"


# ============================================================
# 4. 数据库 Schema
# ============================================================

DATABASE_SCHEMA = """
数据库名称：
marketing_agent_db


1. brands

字段：
- brand_id INT PRIMARY KEY
- brand_name VARCHAR
- category VARCHAR
- official_account VARCHAR


2. creators

字段：
- creator_id INT PRIMARY KEY
- creator_name VARCHAR
- platform VARCHAR
- followers INT
- creator_type VARCHAR
- category VARCHAR


3. posts

字段：
- post_id INT PRIMARY KEY
- platform VARCHAR
- title VARCHAR
- publish_date DATETIME
- likes INT
- comments INT
- favorites INT
- shares INT
- views INT
- brand_id INT
- creator_id INT
- is_viral BOOLEAN

关系：

posts.brand_id
→ brands.brand_id

posts.creator_id
→ creators.creator_id


4. topics

字段：
- topic_id INT PRIMARY KEY
- topic_name VARCHAR


5. post_topics

字段：
- post_id INT
- topic_id INT

关系：

post_topics.post_id
→ posts.post_id

post_topics.topic_id
→ topics.topic_id


6. campaigns

字段：
- campaign_id INT PRIMARY KEY
- campaign_name VARCHAR
- brand_id INT
- start_date DATE
- end_date DATE
- budget DECIMAL

关系：

campaigns.brand_id
→ brands.brand_id


7. creator_performance

字段：
- performance_id INT PRIMARY KEY
- creator_id INT
- campaign_id INT
- post_id INT
- views INT
- likes INT
- comments INT
- favorites INT
- clicks INT
- conversions INT
- cost DECIMAL
- roi DECIMAL

关系：

creator_performance.creator_id
→ creators.creator_id

creator_performance.campaign_id
→ campaigns.campaign_id

creator_performance.post_id
→ posts.post_id
"""


# ============================================================
# 5. HTTP 调用 DeepSeek
# ============================================================

def call_deepseek(messages):
    """
    直接通过 HTTP POST 调用 DeepSeek。
    """

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
    }

    response = requests.post(
        DEEPSEEK_CHAT_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]


# ============================================================
# 6. 让 LLM 生成 SQL
# ============================================================

def generate_sql(question: str):
    """
    把用户自然语言问题转换成 MySQL SELECT 查询。
    """

    system_prompt = f"""
你是一个专业的 MySQL 查询生成器。

你的任务是：

根据用户的问题，
以及下面提供的数据库 Schema，
生成一条正确的 MySQL SELECT 查询。


====================
数据库 Schema
====================

{DATABASE_SCHEMA}


====================
严格规则
====================

1. 只能生成 SELECT 查询。

2. 严禁生成：

INSERT
UPDATE
DELETE
DROP
ALTER
TRUNCATE
CREATE
REPLACE
GRANT
REVOKE

3. 不允许修改数据库中的任何数据。

4. 只能使用 Schema 中真实存在的：
- Table
- Column
- Relationship

5. 如果需要品牌名称：
使用 brands。

6. 如果需要达人名称：
使用 creators。

7. 如果需要帖子数据：
使用 posts。

8. 如果需要主题：
使用 topics + post_topics。

9. 如果需要投放表现：
使用 creator_performance。

10. 如果需要 Campaign：
使用 campaigns。

11. 最近30天应该使用：

DATE_SUB(CURDATE(), INTERVAL 30 DAY)

12. 如果用户要求排名，
合理使用：

ORDER BY
LIMIT

13. 如果需要统计，
合理使用：

COUNT
AVG
SUM
MAX
MIN
GROUP BY

14. 如果计算互动量，默认：

likes + comments + favorites + shares

15. 如果计算帖子互动率，默认：

(likes + comments + favorites + shares) / followers

需要 JOIN creators 获取 followers。

使用 NULLIF(followers, 0)
避免除以 0。

16. 只输出 SQL。

不要解释。

不要输出 Markdown。

不要输出 ```sql。

只返回可以直接执行的 SQL。
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    sql = call_deepseek(messages)

    return clean_sql(sql)


# ============================================================
# 7. 清理模型生成的 SQL
# ============================================================

def clean_sql(sql: str):
    """
    清除 ```sql 等 Markdown 标记。
    """

    sql = sql.strip()

    sql = re.sub(
        r"^```sql",
        "",
        sql,
        flags=re.IGNORECASE,
    )

    sql = re.sub(
        r"^```",
        "",
        sql,
    )

    sql = re.sub(
        r"```$",
        "",
        sql,
    )

    return sql.strip()


# ============================================================
# 8. SQL 安全检查
# ============================================================

FORBIDDEN_SQL_KEYWORDS = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "REPLACE",
    "GRANT",
    "REVOKE",
]


def validate_sql(sql: str):
    """
    检查 SQL 是否是安全的只读查询。

    校验通过：
    返回 True

    校验失败：
    抛出 ValueError
    """

    if not sql:
        raise ValueError(
            "SQL 为空，已拒绝执行。"
        )

    normalized_sql = sql.strip().upper()

    # --------------------------------------------------------
    # 1. 必须 SELECT 开头
    # --------------------------------------------------------

    if not normalized_sql.startswith("SELECT"):
        raise ValueError(
            "只允许执行 SELECT 查询。"
        )

    # --------------------------------------------------------
    # 2. 禁止危险关键词
    # --------------------------------------------------------

    for keyword in FORBIDDEN_SQL_KEYWORDS:

        pattern = rf"\b{keyword}\b"

        if re.search(
            pattern,
            normalized_sql,
        ):

            raise ValueError(
                f"SQL 包含禁止关键词：{keyword}"
            )

    # --------------------------------------------------------
    # 3. 禁止多条 SQL
    # --------------------------------------------------------

    sql_without_final_semicolon = (
        sql.strip().rstrip(";")
    )

    if ";" in sql_without_final_semicolon:

        raise ValueError(
            "检测到多条 SQL，只允许执行单条 SELECT。"
        )

    return True


# ============================================================
# 9. MySQL 查询
# ============================================================

def query_mysql(sql: str):
    """
    执行 SELECT SQL。

    dictionary=True：

    {
        "brand_name": "元气森林",
        "likes": 18000
    }

    更适合外层 Agent 阅读。
    """

    conn = None
    cursor = None

    try:

        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
        )

        cursor = conn.cursor(
            dictionary=True
        )

        cursor.execute(sql)

        rows = cursor.fetchall()

        return rows

    finally:

        if cursor:
            cursor.close()

        if conn:
            conn.close()