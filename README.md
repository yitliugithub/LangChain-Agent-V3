# Research Insight Agent

这是一个面向小红书、抖音广告业务场景的课程项目。它帮助准备在抖音进行广告投放的品牌方，把内部行业报告、项目数据库、公开互联网信息和抖音指数数据组织成可追溯的研究结论，并支持生成品牌研究报告。

这份 README 同时是一份面试复习材料。内容以当前代码为准，并明确区分已经实现的能力和未来可以扩展的方向。

## 1. 项目目标

Research Insight Agent 主要回答四类问题：

1. 行业宏观洞察：行业格局、消费者偏好、营销趋势和历史研究结论。
2. 抖音平台洞察：关键词指数、关联词、人群地域、年龄、性别、兴趣与 TGI。
3. 业务数据分析：品牌、帖子、达人、Campaign、互动量、成本和 ROI。
4. 综合研究报告：结合品牌任务、内部报告和已采集的抖音数据生成中文报告。

当前项目是可解释的本地 Demo，不是生产系统。设计优先级是：能够运行、流程清楚、数据来源可追溯、便于学习和展示。

## 2. 主要功能

- CLI 多轮对话，支持 `new` 清空上下文、`quit` 退出。
- Query Router 将问题分为 `rag`、`web`、`rag_and_web`、`sql`、`douyin`、`report`、`direct` 或 `clarify`。
- 手写、受控的 Agent Loop，限制最多 6 步，避免无限调用工具。
- 基于 Magic-PDF / MinerU 本地 CLI 的 PDF 到 Markdown 预处理。
- Structure-aware + semantic sentence boundary 的 token 分块。
- E5 Dense Retrieval + BM25/Jieba Sparse Retrieval + RRF 融合。
- `BAAI/bge-reranker-v2-m3` Cross-Encoder 重排，最终返回 Top 3。
- Evidence Sufficiency 判断和强制知识库来源引用。
- DeepSeek 生成只读 SQL，并对 SQL 进行校验后查询 MySQL。
- Tavily Web Search 获取实时公开信息。
- 在用户手动登录的浏览器会话中半自动采集抖音指数。
- 通过本地 Skill 约束意图识别和报告生成行为。

## 3. System Overview

```text
                         +----------------------+
User / research_brief -> | app_http.py CLI      |
                         +----------+-----------+
                                    |
                             route_query()
                                    |
       +-------------+--------------+--------------+-------------+
       |             |              |              |             |
     direct        RAG             Web            SQL          Douyin
 calculator   internal reports   Tavily API   MySQL SELECT   local JSON
 current time       |                                              |
                    |                                        collect/update
                    |                                              |
              Hybrid Retrieval                         logged-in Playwright
                    |                                              |
                    +----------------+-----------------------------+
                                     |
                                DeepSeek
                                     |
                         answer with evidence/source

research_brief.json + RAG evidence + Douyin evidence
                         |
                  Report Skill workflow
                         |
              Markdown report + context JSON
```

系统不是让一个模型自由决定并执行所有事情。Router 先限定本轮的数据源和允许使用的工具，Agent Loop 再在这个范围内完成调用。这种设计比完全开放的 Agent 更容易解释、测试和控制。

## 4. Agent 架构

### 4.1 Query Router

`route_query()` 是每轮用户输入的第一步。它加载 `query-intent-router` Skill，让 DeepSeek 返回固定 JSON，然后使用 Pydantic 和额外规则校验。

```json
{
  "route": "rag",
  "action": "answer",
  "needs_clarification": false,
  "clarification_question": null,
  "original_query": "广告主营销投资趋势是什么？",
  "retrieval_query": "广告主营销投资趋势与变化",
  "keywords": [],
  "brief_path": null,
  "reason": "问题需要查询内部行业研究报告"
}
```

各路由的边界：

| Route | 使用场景 |
|---|---|
| `rag` | 内部报告、白皮书、消费者洞察和报告中的统计数据 |
| `web` | 最新新闻、实时事件和会变化的公开信息 |
| `rag_and_web` | 明确要求结合内部研究与当前互联网信息 |
| `sql` | MySQL 中的品牌、帖子、达人、Campaign、表现和 ROI |
| `douyin` | 已采集的抖音指数、关联词和人群数据 |
| `report` | 根据上传的 Excel 或内部 `research_brief.json` 生成完整研究报告 |
| `direct` | 稳定通用知识、数学计算和当前时间 |
| `clarify` | 缺少主体、关键词、研究任务或关键比较条件 |

Router 还会产生一个受约束的 `retrieval_query`。改写只用于检索，最终回答仍围绕原始问题；专有名词、数字、年份和限制条件不能被模型擅自改变。

如果 Router 输出无法解析或未通过校验，当前实现会回退到原始 Agent，让系统仍可继续运行。

### 4.2 手写 Agent Loop

`run_agent()` 的工作过程：

```text
messages -> DeepSeek -> 是否有 tool_calls?
                         |
                 +-------+-------+
                 |               |
                否               是
                 |               |
             最终答案       追加 AI tool call
                                 |
                            执行 Python Tool
                                 |
                            追加 ToolMessage
                                 |
                              再问模型
```

这里虽然使用 `langchain_deepseek.ChatDeepSeek` 作为模型客户端，但没有使用 LangChain Agent Executor 或 LangGraph。工具循环、路由限制、消息追加和最大步数仍由项目代码控制。

`run_agent()` 会把 assistant tool call、tool result 和最终 assistant answer 都写入同一个 `messages` 列表，因此 CLI 不会再次重复追加最终回答。

### 4.3 Tools

| Tool | 输入 | 输出 |
|---|---|---|
| `calculator` | 运算类型和两个数字 | 计算结果 |
| `get_current_time` | 无 | 本机日期时间 |
| `web_search` | 查询文本 | Tavily 搜索结果 |
| `search_knowledge_base` | 检索问题 | Top-K RAG evidence 与充分性判断 |
| `query_marketing_database` | 自然语言问题 | SQL、行数和真实数据库行 |
| `query_douyin_index_data` | 关键词列表 | 本地抖音采集状态与结构化数据 |

## 5. RAG 架构

### 5.1 离线构建

```text
PDF files
   |
   v
Magic-PDF / MinerU
   |
   v
cleaned Markdown
   |
   v
Heading -> paragraph/table -> sentence -> token fallback
   |
   v
Structure-aware + semantic boundary chunks (500/150 tokens)
   |
   v
multilingual-e5-base passage embeddings (normalized)
   |
   v
Chroma PersistentClient / marketing_knowledge
```

### 5.2 为什么 PDF 先转 Markdown

PDF 更像页面绘制格式，不天然代表正确阅读顺序。复杂报告可能有双栏、页眉页脚、图表和表格。Magic-PDF 先恢复标题、段落、表格和图片引用，Markdown 再为后续分块提供可识别的结构边界。

当前正式入口使用 Magic-PDF / MinerU 本地 CLI。PP-Structure 和 PyMuPDF4LLM 的代码及结果属于解析方案对比实验，不是正式预处理基线。

### 5.3 分块策略

冻结参数：

```text
max_tokens = 500
min_tokens = 150
method = structure-aware + semantic sentence boundary
```

分块优先级：

1. Markdown heading：先按标题划分 Section。
2. Paragraph/Table：保留段落空行和 Markdown 表格边界。
3. Sentence：普通段落按句子拆成基本单元。
4. Semantic boundary：chunk 超长时，优先在相邻单元语义距离较大的位置切分。
5. Token fallback：单句或单行仍超过上限时才按 token 兜底。

每个 chunk 前加入 `Section: ...`，让离开原文位置的片段仍保留章节语境。这里的 Section 来自预处理 Markdown 的标题。

500/150 不是通用最佳值，而是当前语料和评测集上的 prototype baseline。500 控制上下文完整性与检索粒度的平衡，150 用于减少过短且信息不足的片段。未来换语料后仍应重新评测。

Chunk metadata 包括：

- `source`
- `source_file`
- `section`
- `chunk_index`
- `token_count`
- `chunk_method`
- `block_types`

Chunk ID 使用来源、Section、序号和内容哈希生成。输入内容不变时 ID 可稳定复现。

### 5.4 Embedding 与 ChromaDB

当前模型是 `intfloat/multilingual-e5-base`。

- 文档使用 `passage: ` 前缀。
- 查询使用 `query: ` 前缀。
- 构建端和查询端都进行 L2 normalization。
- Chroma collection 显式使用 cosine distance。

ChromaDB 的角色是保存：

```text
chunk ID + chunk text + embedding vector + metadata
```

查询时输入 query embedding，Chroma 返回最相近的 chunk ID 和 cosine distance。它负责向量存储与近邻搜索，不负责生成答案。

### 5.5 在线混合检索

冻结检索基线：

```text
Dense Top 10
BM25 + Jieba Top 5
RRF_K = 60
Candidate chunks -> bge-reranker-v2-m3
Final Top 3
```

流程：

```text
Query
  +--> E5 Dense Search -------- Top 10 --+
  |                                      |
  +--> Jieba + BM25 Search ---- Top 5 ---+--> RRF --> Reranker --> Top 3
```

- Dense Retrieval 擅长召回语义相近但用词不同的内容。
- BM25 擅长匹配品牌名、产品名、数字和专有词。
- Jieba 为中文 BM25 提供分词，自定义词典避免领域词被错误切开。
- RRF 使用排名而不是直接混合两种不可比的原始分数。
- Cross-Encoder 同时读取 query 和候选 chunk，判断它们的细粒度相关性并重新排序。

RRF 分数为：

```text
RRF(d) = sum(1 / (60 + rank_i(d)))
```

RRF 的 `60` 是平滑常数，不是 Top-K，也不是相关性阈值。它降低第一名对融合结果的过度支配。

### 5.6 Evidence Sufficiency 与引用

检索相关不等于证据足够。`check_evidence_sufficiency()` 把原始问题、改写问题和 Top 3 evidence 交给 DeepSeek 判断：

- `answer`：证据足够，可以回答。
- `web`：内部证据不足，但缺失信息适合由公开互联网补充。
- `answer_insufficient`：缺少内部事实，明确告诉用户证据不足。

判断失败时采用保守回退，不会假装证据充分。RAG 回答末尾会追加报告名、Section 和 Chunk 编号。

## 6. 抖音指数采集

抖音创作者中心没有在本项目中使用公开数据 API。当前采用半自动方案：用户在专用 Chrome profile 中手动登录，Playwright 在这个已登录会话里逐个采集审核通过的关键词。

采集模块：

- 关键词指数
- 关联分析
- 人群分析：地域、年龄、性别、兴趣和 TGI
- 页面原始文本
- 分段截图与滚动长截图
- 每次采集的状态、时间范围和来源 URL

周期规则：

- 关键词指数和人群分析支持 `7d`、`14d`、`30d`、`6m`。
- 关联分析由平台固定为最近 7 天。
- 如果关键词未被平台收录，记录为 `skipped / keyword_not_indexed` 并继续下一个词；不能把未收录解释为热度为零。

采集结果使用 JSON，是因为字段、状态、周期、数字、URL 和图片路径可以被程序稳定读取；滚动截图作为报告视觉证据保留，但不能替代结构化数字。

## 7. SQL Tool

```text
Natural-language question
        |
generate_sql() -> DeepSeek HTTP POST
        |
clean_sql() -> remove Markdown fences
        |
validate_sql()
        |
query_mysql() -> raw rows
        |
outer DeepSeek -> natural-language answer
```

安全规则：

- 只允许单条 `SELECT`。
- 禁止注释和多条 SQL。
- 禁止 `INSERT`、`UPDATE`、`DELETE`、`DROP`、`ALTER` 等写操作。
- 只允许 Schema 中声明的表。
- Cursor 和 Connection 在 `finally` 中关闭。

需要诚实说明：当前校验属于 prototype 防护，不是完整 SQL parser，也没有数据库级只读账户配置代码。生产环境仍应使用只读 MySQL 用户、查询超时、行数上限和成熟 SQL parser。

## 8. Skill 与报告生成

Skill 是保存在 `SKILL.md` 中的一组本地行为规范。它负责告诉模型何时使用某个流程、必须遵守什么规则以及输出格式；Python 模块负责真正执行数据读取、检索和模型调用。

当前有两个 Skill：

| Skill | 职责 |
|---|---|
| `query-intent-router` | 识别用户意图并选择数据源，不生成业务答案 |
| `douyin-research-report` | 约束完整品牌研究报告的证据、章节、引用和图片规则 |

报告流程：

```text
research_input.xlsx
        |
program-generated keyword candidates
        |
human keyword review (approved=yes)
        |
research_brief.json
        +--> latest local Douyin JSON and screenshots
        +--> internal RAG evidence
        +--> report Skill instructions
        |
      DeepSeek
        |
Markdown report + report_context.json
```

`report_context.json` 保存生成报告时使用的任务和证据，便于检查报告结论来自哪里。

## 9. 关键文件职责

当前目录结构：

```text
pythonProject2/
├── app_http.py                 # CLI 主入口
├── preprocess_pdf.py           # 正式 PDF 预处理入口
├── build_rag.py                # 正式索引构建
├── search_rag.py               # 正式 RAG 检索
├── sql_agent.py                # SQL 工具
├── rag_config.py               # RAG 路径与冻结参数
├── rag_models.py               # E5 与 reranker
├── report_skill.py             # 报告生成
├── skill_loader.py             # Skill 加载
├── tools/                      # 采集、Excel、词典和批处理工具
├── experiments/                # 按研究阶段归档的实验代码
├── evaluation/                 # 数据集、候选集、结果和评测脚本
├── data/                       # 原始 PDF、标准 Markdown、抖音数据
├── artifacts/                  # PDF/分块实验产物和历史报告
├── storage/chroma_db/          # Chroma 持久化向量库
├── resources/jieba_dictionary/
├── skills/
├── notebooks/
└── uploads/                    # 用户放置品牌研究 Excel
```

| 文件 | 职责 |
|---|---|
| `app_http.py` | CLI、Router、工具 Schema、Agent Loop、对话记忆、证据充分性和引用 |
| `preprocess_pdf.py` | 正式 PDF 预处理入口，将解析结果同步为标准 Markdown |
| `experiments/pdf_parsing/experiment_magic_pdf.py` | Magic-PDF 执行与清洗实现，供正式预处理和实验复用 |
| `build_rag.py` | 分块、文档 embedding、metadata 和 Chroma collection 重建 |
| `search_rag.py` | Dense、BM25/Jieba、RRF、reranker 和 Top-K 返回 |
| `rag_config.py` | 冻结的路径、模型和检索参数 |
| `rag_models.py` | E5 embedding 和 Cross-Encoder reranker |
| `sql_agent.py` | SQL 生成、清洗、校验和 MySQL 查询 |
| `tools/collect_douyin_index.py` | 抖音半自动采集器，一次处理全部审核词 |
| `tools/prepare_brand_research_input.py` | 校验 Excel、生成关键词审核表并导出 Agent JSON |
| `report_skill.py` | 汇总任务、RAG、抖音数据并执行报告生成工作流 |
| `skill_loader.py` | 读取和校验本地 `SKILL.md` |
| `skills/` | Router 与报告的行为说明和格式规范 |
| `evaluation/datasets/` | 人工审核后的评测集 |
| `evaluation/results/` | Dense、BM25、Hybrid、Reranker 和 Query Handling 实验结果 |
| `uploads/` | 用户上传或放置研究输入 Excel 的固定目录 |

所有 `experiment_*.py` 和实验输出用于证明技术选择过程，不属于在线 Agent 主链路。

## 10. 安装与配置

建议使用 Python 3.10 环境。项目曾使用 `myenv` 运行 Agent/RAG，Magic-PDF 可以放在独立的 `mineru_magic` 环境中。

安装主项目依赖：

```bash
pip install -r requirements.txt
playwright install chrome
```

Magic-PDF 按官方说明单独安装，zsh 中必须给 extras 加引号：

```bash
pip install -U 'magic-pdf[full]' --extra-index-url https://wheels.myhloli.com
```

并确保官方模型下载脚本已经生成：

```text
~/magic-pdf.json
```

`.env` 使用以下变量，不能提交到 Git：

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=
TAVILY_API_KEY=
MYSQL_HOST=
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_DATABASE=
MinerU_API_KEY=
```

当前正式预处理走本地 Magic-PDF CLI，不需要 `MinerU_API_KEY`；该变量是早期 MinerU API 实验遗留配置。

## 11. 运行流程

### 11.1 构建 RAG

将 PDF 放进 `data/raw_pdfs/`：

```bash
python preprocess_pdf.py
python build_rag.py
```

只处理一个 PDF：

```bash
python preprocess_pdf.py --pdf "data/raw_pdfs/example.pdf"
```

直接测试检索：

```bash
python search_rag.py
```

### 11.2 启动 Agent

```bash
python app_http.py
```

CLI 命令：

```text
new                              清空当前对话
skills                           查看可用 Skill
report <研究输入.xlsx>           校验输入、生成关键词并在审核后生成报告
report <research_brief.json>     兼容旧流程，直接生成完整研究报告
quit                             退出
```

### 11.3 品牌研究输入和抖音采集

推荐直接把 Excel 交给 Agent：

```text
python app_http.py
User: report uploads/research_input_template.xlsx
```

首次提交时，程序校验 `research_brief` 并把候选词写入 `keyword_review`，然后停止等待人工审核。
把采用项设置为 `yes`、不采用项设置为 `no` 后，再次输入同一个 `report` 命令。程序会自动
生成内部 `research_brief.json`，检查每个 approved 关键词是否有相同时间周期的采集结果；
缺失时会启动一次可见 Chrome，在同一个登录会话中顺序采集最多 10 个关键词。只有采集状态
通过检查后才执行报告 Skill。
JSON 是可追溯的中间数据，用户不需要手工编写。

也可以独立运行转换脚本。第一次生成关键词，审核后第二次导出 JSON：

```bash
python tools/prepare_brand_research_input.py uploads/research_input.xlsx
```

一次采集全部审核通过的关键词：

```bash
python tools/collect_douyin_index.py --brief uploads/research_brief.json --all-approved
```

`--batch-number` 和 `--batch-count` 仅保留给采集实验，正式报告流程不使用分批模式。

测试单个关键词和指定周期：

```bash
python tools/collect_douyin_index.py --keyword 乐事 --time-range 14d
```

归档后的实验和评测脚本应在项目根目录用模块方式运行，例如：

```bash
python -m experiments.retrieval.experiment_dense_only_evaluation
python -m evaluation.scripts.validate_evaluation_set
```

这样 Python 会从项目根目录解析 `experiments`、`evaluation` 和正式 RAG 模块；不要在子目录中直接运行这些脚本文件。

生成报告：

```text
python app_http.py
User: report uploads/research_input.xlsx
```

## 12. 面试重点理解

### 为什么不是只用向量检索？

营销报告同时包含语义问题和品牌、数字、平台等精确词。Dense 擅长语义，BM25 擅长词面匹配，混合后覆盖更稳。

### 为什么需要 RRF？

Cosine similarity 与 BM25 score 的量纲不同，不能直接相加。RRF 只使用名次进行融合，不依赖原始分数尺度。

### 为什么 RRF 后还要 Reranker？

前两路检索负责高召回，Cross-Encoder 负责更精细地判断 query 与候选文本是否真正相关，从而改善最终 Top 3 的顺序。

### 为什么使用 Top 3？

最终 Top 3 是当前评测集上的冻结 baseline。候选阶段已经使用 Dense 10 和 BM25 5 扩大召回，最终阶段压缩到 3 条，以降低无关上下文和模型输入成本。它不是所有数据集的固定最佳值。

### 为什么既有 Router 又有 Agent Tool Calling？

Router 先决定允许使用的数据源，减少错误工具调用；Agent Loop 负责执行被允许的工具并组织答案。前者控制边界，后者完成动作。

### Evidence Sufficiency 解决什么问题？

Top 3 可能与问题相关，但不一定覆盖全部数字、年份或比较对象。充分性判断阻止模型把不完整证据包装成确定答案。

### ChromaDB 在 RAG 中做什么？

它保存 chunk、embedding 和 metadata，并根据 query embedding 返回相近片段。它是检索存储层，不是 LLM，也不生成答案。

### 项目使用 LangChain 吗？

使用了 `langchain_deepseek.ChatDeepSeek` 和 LangChain message 类型作为模型调用层；没有使用 LangChain Agent 或 LangGraph。核心 Router、工具执行和 Agent Loop 都是手写的。`sql_agent.py` 当前仍通过 `requests.post()` 直接调用 DeepSeek HTTP API。

## 13. 已知限制

- 抖音采集依赖页面 DOM、有效登录会话和平台权限，页面改版可能导致 selector 失效。
- 抖音采集是用户触发的半自动流程，不是后台无人值守任务。
- 报告目前输出 Markdown；图片以本地路径插入，还没有稳定导出为 PDF 或 Word。
- 报告流程一次只接受一个 task；缺少匹配周期的抖音数据时会启动可见浏览器采集。
- 当前 RAG 主要处理文本和 Markdown 表格；图片被保留，但没有完整的视觉理解或多模态向量检索。
- 当前没有 Knowledge Graph 或 Graph RAG。参考图中的图检索不是本项目已经实现的功能。
- BM25 索引在 Retriever 初始化时从 Chroma 全量文档加载并在内存构建，适合当前小型 Demo，不适合超大知识库。
- RAG、Router 和充分性判断的效果受现有评测集规模与质量限制。
- LLM-as-judge 可以提高标注效率，但存在模型偏差，关键样本仍应人工抽查。
- SQL 安全校验是规则式 prototype，生产环境需要数据库只读账号、查询限时和更严格的解析器。
- `preprocess_pdf.py` 仍复用 PDF 解析实验中的 Magic-PDF 实现；若未来继续产品化，可再将稳定解析逻辑提取到 `tools/`。

## 14. 当前 Baseline

```text
PDF parser: Magic-PDF / MinerU local CLI
Chunking: structure-aware + semantic sentence boundary
Chunk size: max 500 tokens / min 150 tokens
Embedding: intfloat/multilingual-e5-base
Dense candidates: 10
Sparse candidates: BM25 + Jieba, 5
Fusion: RRF, k=60
Reranker: BAAI/bge-reranker-v2-m3
Final evidence: Top 3
LLM: DeepSeek through langchain_deepseek.ChatDeepSeek
Agent control: validated Router + hand-written loop, max 6 steps
```

Baseline 的含义是“当前实验后冻结、用于 Demo 的配置”，不是宣称这些参数对所有文档和业务场景都是最佳值。
