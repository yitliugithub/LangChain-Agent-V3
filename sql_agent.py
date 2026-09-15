import os
import re

import requests
from dotenv import load_dotenv


load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

MODEL_NAME = "deepseek-flash"
DEEPSEEK_CHAT_URL = (
    DEEPSEEK_BASE_URL.rstrip("/") + "/chat/completions"
    if DEEPSEEK_BASE_URL
    else ""
)


DATABASE_SCHEMA = """
数据库名称：
marketing_agent_db

1. brands
- brand_id INT PRIMARY KEY
- brand_name VARCHAR
- category VARCHAR
- official_account VARCHAR

2. creators
- creator_id INT PRIMARY KEY
- creator_name VARCHAR
- platform VARCHAR
- followers INT
- creator_type VARCHAR
- category VARCHAR

3. posts
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
posts.brand_id -> brands.brand_id
posts.creator_id -> creators.creator_id

4. topics
- topic_id INT PRIMARY KEY
- topic_name VARCHAR

5. post_topics
- post_id INT
- topic_id INT

关系：
post_topics.post_id -> posts.post_id
post_topics.topic_id -> topics.topic_id

6. campaigns
- campaign_id INT PRIMARY KEY
- campaign_name VARCHAR
- brand_id INT
- start_date DATE
- end_date DATE
- budget DECIMAL

关系：
campaigns.brand_id -> brands.brand_id

7. creator_performance
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
creator_performance.creator_id -> creators.creator_id
creator_performance.campaign_id -> campaigns.campaign_id
creator_performance.post_id -> posts.post_id
"""


ALLOWED_TABLES = {
    "brands",
    "creators",
    "posts",
    "topics",
    "post_topics",
    "campaigns",
    "creator_performance",
}

FORBIDDEN_SQL_KEYWORDS = {
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
}


def require_env(keys: list[str]) -> None:
    values = {
        "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
        "DEEPSEEK_BASE_URL": DEEPSEEK_BASE_URL,
        "MYSQL_HOST": MYSQL_HOST,
        "MYSQL_USER": MYSQL_USER,
        "MYSQL_PASSWORD": MYSQL_PASSWORD,
        "MYSQL_DATABASE": MYSQL_DATABASE,
    }
    missing = [key for key in keys if not values.get(key)]
    if missing:
        raise ValueError("缺少环境变量：" + ", ".join(missing))


def call_deepseek(messages):
    require_env(["DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"])

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


def generate_sql(question: str):
    system_prompt = f"""
你是一个专业的 MySQL 查询生成器。

你的任务是根据用户问题和数据库 Schema 生成一条安全、正确的 MySQL SELECT 查询。

====================
数据库 Schema
====================
{DATABASE_SCHEMA}

====================
严格规则
====================
1. 只能生成一条 SELECT 查询。
2. 严禁生成 INSERT、UPDATE、DELETE、DROP、ALTER、TRUNCATE、CREATE、REPLACE、GRANT、REVOKE。
3. 不允许修改数据库中的任何数据。
4. 只能使用 Schema 中真实存在的 table、column 和 relationship。
5. 如果需要品牌名称，使用 brands。
6. 如果需要达人名称，使用 creators。
7. 如果需要帖子数据，使用 posts。
8. 如果需要主题，使用 topics + post_topics。
9. 如果需要投放表现，使用 creator_performance。
10. 如果需要 Campaign，使用 campaigns。
11. 最近30天使用 DATE_SUB(CURDATE(), INTERVAL 30 DAY)。
12. 排名问题合理使用 ORDER BY 和 LIMIT。
13. 统计问题合理使用 COUNT、AVG、SUM、MAX、MIN、GROUP BY。
14. 互动量默认是 likes + comments + favorites + shares。
15. 帖子互动率默认是 (likes + comments + favorites + shares) / followers，需要 JOIN creators 并使用 NULLIF(followers, 0)。
16. 只输出 SQL，不要解释，不要 Markdown，不要 ```sql。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    sql = call_deepseek(messages)
    return clean_sql(sql)


def clean_sql(sql: str):
    if not sql:
        return ""

    cleaned = sql.strip()
    cleaned = re.sub(r"^```(?:sql|mysql)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    match = re.search(r"\bSELECT\b", cleaned, flags=re.IGNORECASE)
    if match:
        cleaned = cleaned[match.start():].strip()

    return cleaned


def _sql_without_strings(sql: str) -> str:
    sql = re.sub(r"'(?:''|[^'])*'", "''", sql)
    sql = re.sub(r'"(?:""|[^"])*"', '""', sql)
    return sql


def validate_sql(sql: str):
    if not sql:
        raise ValueError("SQL 为空，已拒绝执行。")

    stripped_sql = sql.strip()
    normalized_sql = _sql_without_strings(stripped_sql).upper()

    if not normalized_sql.startswith("SELECT"):
        raise ValueError("只允许执行 SELECT 查询。")

    if "--" in normalized_sql or "/*" in normalized_sql or "*/" in normalized_sql:
        raise ValueError("SQL 中不允许包含注释。")

    sql_without_final_semicolon = stripped_sql.rstrip(";")
    if ";" in sql_without_final_semicolon:
        raise ValueError("检测到多条 SQL，只允许执行单条 SELECT。")

    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", normalized_sql):
            raise ValueError(f"SQL 包含禁止关键词：{keyword}")

    referenced_tables = {
        table.lower()
        for table in re.findall(
            r"\b(?:FROM|JOIN)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?",
            stripped_sql,
            flags=re.IGNORECASE,
        )
    }
    unknown_tables = referenced_tables - ALLOWED_TABLES
    if unknown_tables:
        raise ValueError("SQL 使用了未知表：" + ", ".join(sorted(unknown_tables)))

    return True


def query_mysql(sql: str):
    import mysql.connector

    require_env(["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"])
    validate_sql(sql)

    conn = None
    cursor = None

    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql)
        return cursor.fetchall()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
