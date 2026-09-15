# Query Handling Evaluation Set Review

请把每条的 `review_decision` 改成 `yes` 或 `no`。
需要修改时，可直接编辑 Query、Canonical Query、反问或限制条件，并在 `review_notes` 说明。

## 审核标准

- `clear`：信息完整，应直接检索。
- `expandable`：表达较宽泛，但只有一个合理意图，可以安全改写。
- `ambiguous`：存在多个合理意图，必须反问。
- 改写不得改变数字、实体、年份、比较范围或问题任务。

## 抽样信息

- 源问题：`10`
- 草稿问题：`30`
- 类型分布：`{'explain': 3, 'number': 3, 'fact': 3, 'compare': 1}`
- 文档数：`7`

## QH01-1 · 清晰问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q001`
- **source_document:** 2024年Social&KOL营销趋势报告
- **user_query:** 根据报告，用户为什么开始喜欢去品牌自播环境？
- **expected_route:** `clear`
- **canonical_query:** 根据报告，用户为什么开始喜欢去品牌自播环境？
- **must_preserve:** ["品牌自播", "用户", "原因"]
- **must_not_add:** ["达播", "实惠", "卖点", "折扣", "售后服务"]

---

## QH01-2 · 可扩写问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q001`
- **source_document:** 2024年Social&KOL营销趋势报告
- **user_query:** 报告里说，用户为啥现在更爱去品牌自己的直播间了？
- **expected_route:** `expandable`
- **canonical_query:** 根据报告，用户为什么开始喜欢去品牌自播环境？
- **must_preserve:** ["品牌自播", "用户", "原因"]
- **must_not_add:** ["达播", "实惠", "卖点", "折扣", "售后服务"]

---

## QH01-3 · 模糊问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q001`
- **source_document:** 2024年Social&KOL营销趋势报告
- **user_query:** 用户为什么更喜欢某种直播方式？
- **expected_route:** `ambiguous`
- **canonical_query:** 根据报告，用户为什么开始喜欢去品牌自播环境？
- **must_preserve:** ["品牌自播", "用户", "原因"]
- **must_not_add:** ["达播", "实惠", "卖点", "折扣", "售后服务"]
- **expected_clarifying_question:** 您指的是品牌自播还是达人直播？
- **clarification_slots:** ["直播类型（品牌自播或达人直播）"]

---

## QH02-1 · 清晰问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q002`
- **source_document:** 2024年Social&KOL营销趋势报告
- **user_query:** 根据报告，在小红书被美妆护肤香水品类种草后，有多大比例的用户会点击链接查看详情？
- **expected_route:** `clear`
- **canonical_query:** 根据报告，在小红书被美妆护肤香水品类种草后，有多大比例的用户会点击链接查看详情？
- **must_preserve:** ["小红书", "种草", "点击链接", "美妆护肤香水品类"]
- **must_not_add:** ["具体品牌", "时间范围", "其他平台"]

---

## QH02-2 · 可扩写问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q002`
- **source_document:** 2024年Social&KOL营销趋势报告
- **user_query:** 报告里说，在小红书上看了美妆护肤香水的内容后，有多少人会点链接看详情？
- **expected_route:** `expandable`
- **canonical_query:** 根据报告，在小红书被美妆护肤香水品类种草后，有多大比例的用户会点击链接查看详情？
- **must_preserve:** ["小红书", "种草", "点击链接", "美妆护肤香水品类"]
- **must_not_add:** ["具体品牌", "时间范围", "其他平台"]

---

## QH02-3 · 模糊问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q002`
- **source_document:** 2024年Social&KOL营销趋势报告
- **user_query:** 在小红书被种草后，用户会点击链接查看详情吗？
- **expected_route:** `ambiguous`
- **canonical_query:** 根据报告，在小红书被美妆护肤香水品类种草后，有多大比例的用户会点击链接查看详情？
- **must_preserve:** ["小红书", "种草", "点击链接", "美妆护肤香水品类"]
- **must_not_add:** ["具体品牌", "时间范围", "其他平台"]
- **expected_clarifying_question:** 您指的是哪个品类？以及您想了解点击链接的用户比例还是其他行为？
- **clarification_slots:** ["品类", "具体指标（如比例）"]

---

## QH03-1 · 清晰问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q010`
- **source_document:** 2025中国消费者生活方式演进趋势系列报告——酒水饮料篇-省广集团
- **user_query:** 根据报告，2011年至今酒类市场在味型上有哪些创新趋势？
- **expected_route:** `clear`
- **canonical_query:** 根据报告，2011年至今酒类市场在味型上有哪些创新趋势？
- **must_preserve:** ["2011年至今", "酒类市场", "味型", "创新趋势"]
- **must_not_add:** ["具体香型如豉香、陶香", "具体产品如葛根啤酒", "具体趋势如个性化、本土化"]

---

## QH03-2 · 可扩写问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q010`
- **source_document:** 2025中国消费者生活方式演进趋势系列报告——酒水饮料篇-省广集团
- **user_query:** 报告里说从2011年到现在，酒类口味方面有哪些新的发展方向？
- **expected_route:** `expandable`
- **canonical_query:** 根据报告，2011年至今酒类市场在味型上有哪些创新趋势？
- **must_preserve:** ["2011年至今", "酒类市场", "味型", "创新趋势"]
- **must_not_add:** ["具体香型如豉香、陶香", "具体产品如葛根啤酒", "具体趋势如个性化、本土化"]

---

## QH03-3 · 模糊问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q010`
- **source_document:** 2025中国消费者生活方式演进趋势系列报告——酒水饮料篇-省广集团
- **user_query:** 酒类市场有哪些创新趋势？
- **expected_route:** `ambiguous`
- **canonical_query:** 根据报告，2011年至今酒类市场在味型上有哪些创新趋势？
- **must_preserve:** ["2011年至今", "酒类市场", "味型", "创新趋势"]
- **must_not_add:** ["具体香型如豉香、陶香", "具体产品如葛根啤酒", "具体趋势如个性化、本土化"]
- **expected_clarifying_question:** 您指的是哪个时间范围？以及您关注的是味型还是其他方面的创新趋势？
- **clarification_slots:** ["时间范围", "创新趋势的具体方面（如味型、包装、营销等）"]

---

## QH04-1 · 清晰问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q012`
- **source_document:** 2025中国量贩零食行业现状报告
- **user_query:** 万辰集团的门店数量从2022年到2024年7月增长了多少家？
- **expected_route:** `clear`
- **canonical_query:** 万辰集团的门店数量从2022年到2024年7月增长了多少家？
- **must_preserve:** ["万辰集团", "门店数量", "增长", "2022年", "2024年7月"]
- **must_not_add:** ["超过7000家", "百余家", "具体增长数字"]

---

## QH04-2 · 可扩写问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q012`
- **source_document:** 2025中国量贩零食行业现状报告
- **user_query:** 万辰集团的门店数量在2022年到2024年7月之间增加了多少？
- **expected_route:** `expandable`
- **canonical_query:** 万辰集团的门店数量从2022年到2024年7月增长了多少家？
- **must_preserve:** ["万辰集团", "门店数量", "增长", "2022年", "2024年7月"]
- **must_not_add:** ["超过7000家", "百余家", "具体增长数字"]

---

## QH04-3 · 模糊问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q012`
- **source_document:** 2025中国量贩零食行业现状报告
- **user_query:** 万辰集团的门店数量增长了多少？
- **expected_route:** `ambiguous`
- **canonical_query:** 万辰集团的门店数量从2022年到2024年7月增长了多少家？
- **must_preserve:** ["万辰集团", "门店数量", "增长", "2022年", "2024年7月"]
- **must_not_add:** ["超过7000家", "百余家", "具体增长数字"]
- **expected_clarifying_question:** 您指的是哪个时间段内的门店数量增长？
- **clarification_slots:** ["起始时间", "结束时间"]

---

## QH05-1 · 清晰问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q013`
- **source_document:** 2025中国量贩零食行业现状报告
- **user_query:** 为什么直营连锁模式扩张速度相对较慢？
- **expected_route:** `clear`
- **canonical_query:** 为什么说直营连锁模式扩张速度相对较慢？
- **must_preserve:** ["直营连锁", "扩张速度", "资金压力"]
- **must_not_add:** ["加盟模式", "具体资金数额", "时间范围"]

---

## QH05-2 · 可扩写问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q013`
- **source_document:** 2025中国量贩零食行业现状报告
- **user_query:** 直营连锁为什么扩张得慢？
- **expected_route:** `expandable`
- **canonical_query:** 为什么说直营连锁模式扩张速度相对较慢？
- **must_preserve:** ["直营连锁", "扩张速度", "资金压力"]
- **must_not_add:** ["加盟模式", "具体资金数额", "时间范围"]

---

## QH05-3 · 模糊问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q013`
- **source_document:** 2025中国量贩零食行业现状报告
- **user_query:** 为什么扩张速度相对较慢？
- **expected_route:** `ambiguous`
- **canonical_query:** 为什么说直营连锁模式扩张速度相对较慢？
- **must_preserve:** ["直营连锁", "扩张速度", "资金压力"]
- **must_not_add:** ["加盟模式", "具体资金数额", "时间范围"]
- **expected_clarifying_question:** 您指的是哪种扩张模式？
- **clarification_slots:** ["扩张模式"]

---

## QH06-1 · 清晰问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q031`
- **source_document:** 2025年中国宠物行业市场报告
- **user_query:** 2024年养宠人在选择宠物主粮品牌时，最看重的因素是什么？
- **expected_route:** `clear`
- **canonical_query:** 2024年养宠人在选择宠物主粮品牌时，最看重的因素是什么？
- **must_preserve:** ["2024年", "养宠人", "选择宠物主粮品牌", "最看重的因素"]
- **must_not_add:** ["具体品牌名称", "价格区间", "营养成分", "宠物种类"]

---

## QH06-2 · 可扩写问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q031`
- **source_document:** 2025年中国宠物行业市场报告
- **user_query:** 2024年大家挑宠物主粮牌子的时候，最在意啥？
- **expected_route:** `expandable`
- **canonical_query:** 2024年养宠人在选择宠物主粮品牌时，最看重的因素是什么？
- **must_preserve:** ["2024年", "养宠人", "选择宠物主粮品牌", "最看重的因素"]
- **must_not_add:** ["具体品牌名称", "价格区间", "营养成分", "宠物种类"]

---

## QH06-3 · 模糊问题

- **review_decision:** `no`
- **review_notes:** `已更改`
- **source_evaluation_id:** `Q031`
- **source_document:** 2025年中国宠物行业市场报告
- **user_query:** 养宠人选粮时最看重什么？
- **expected_route:** `ambiguous`
- **canonical_query:** 2024年养宠人在选择宠物主粮品牌时，最看重的因素是什么？
- **must_preserve:** ["2024年", "养宠人", "选择宠物主粮品牌", "最看重的因素"]
- **must_not_add:** ["具体品牌名称", "价格区间", "营养成分", "宠物种类"]
- **expected_clarifying_question:** 您指的是宠物主粮吗？
- **clarification_slots:** ["年份", "选择对象（主粮品牌）"]

---

## QH07-1 · 清晰问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q032`
- **source_document:** 2025年中国宠物行业市场报告
- **user_query:** 与国际知名品牌相比，中国宠物行业在品牌建设和执行标准上存在哪些不足？
- **expected_route:** `clear`
- **canonical_query:** 与国际知名品牌相比，中国宠物行业在哪些方面存在不足？
- **must_preserve:** ["与国际知名品牌相比", "中国宠物行业", "不足"]
- **must_not_add:** ["具体品牌名称", "具体数据或年份", "其他行业领域"]

---

## QH07-2 · 可扩写问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q032`
- **source_document:** 2025年中国宠物行业市场报告
- **user_query:** 和国际大牌比，中国宠物行业差在哪儿？
- **expected_route:** `expandable`
- **canonical_query:** 与国际知名品牌相比，中国宠物行业在哪些方面存在不足？
- **must_preserve:** ["与国际知名品牌相比", "中国宠物行业", "不足"]
- **must_not_add:** ["具体品牌名称", "具体数据或年份", "其他行业领域"]

---

## QH07-3 · 模糊问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q032`
- **source_document:** 2025年中国宠物行业市场报告
- **user_query:** 中国宠物行业存在哪些不足？
- **expected_route:** `ambiguous`
- **canonical_query:** 与国际知名品牌相比，中国宠物行业在哪些方面存在不足？
- **must_preserve:** ["与国际知名品牌相比", "中国宠物行业", "不足"]
- **must_not_add:** ["具体品牌名称", "具体数据或年份", "其他行业领域"]
- **expected_clarifying_question:** 您想从哪些方面进行比较？例如品牌建设、执行标准，还是其他维度？
- **clarification_slots:** ["比较对象（如国际知名品牌）", "比较维度（如品牌建设、执行标准）"]

---

## QH08-1 · 清晰问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q040`
- **source_document:** 2025年中国银发经济研究报告
- **user_query:** 根据工信部数据，截至2023年，我国养老用品市场规模达到多少？
- **expected_route:** `clear`
- **canonical_query:** 根据工信部数据，截至2023年，我国养老用品市场规模达到多少？
- **must_preserve:** ["工信部数据", "截至2023年", "养老用品", "市场规模"]
- **must_not_add:** ["具体金额", "增长率", "细分领域", "其他年份"]

---

## QH08-2 · 可扩写问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q040`
- **source_document:** 2025年中国银发经济研究报告
- **user_query:** 2023年我国养老用品市场规模有多大？
- **expected_route:** `expandable`
- **canonical_query:** 根据工信部数据，截至2023年，我国养老用品市场规模达到多少？
- **must_preserve:** ["工信部数据", "截至2023年", "养老用品", "市场规模"]
- **must_not_add:** ["具体金额", "增长率", "细分领域", "其他年份"]

---

## QH08-3 · 模糊问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q040`
- **source_document:** 2025年中国银发经济研究报告
- **user_query:** 我国养老用品市场规模有多大？
- **expected_route:** `ambiguous`
- **canonical_query:** 根据工信部数据，截至2023年，我国养老用品市场规模达到多少？
- **must_preserve:** ["工信部数据", "截至2023年", "养老用品", "市场规模"]
- **must_not_add:** ["具体金额", "增长率", "细分领域", "其他年份"]
- **expected_clarifying_question:** 您指的是哪一年的市场规模？
- **clarification_slots:** ["年份"]

---

## QH09-1 · 清晰问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q045`
- **source_document:** 2025年品牌营销趋势报告
- **user_query:** 线下户外投放对品牌广告主有什么作用？
- **expected_route:** `clear`
- **canonical_query:** 线下户外投放对这类品牌广告主有什么作用？
- **must_preserve:** ["线下户外投放", "品牌广告主", "作用"]
- **must_not_add:** ["特定时间地点", "线上成交", "精准广触达", "营销可验证", "提升效率"]

---

## QH09-2 · 可扩写问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q045`
- **source_document:** 2025年品牌营销趋势报告
- **user_query:** 品牌广告主做线下户外广告能带来哪些好处？
- **expected_route:** `expandable`
- **canonical_query:** 线下户外投放对这类品牌广告主有什么作用？
- **must_preserve:** ["线下户外投放", "品牌广告主", "作用"]
- **must_not_add:** ["特定时间地点", "线上成交", "精准广触达", "营销可验证", "提升效率"]

---

## QH09-3 · 模糊问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q045`
- **source_document:** 2025年品牌营销趋势报告
- **user_query:** 线下投放有什么作用？
- **expected_route:** `ambiguous`
- **canonical_query:** 线下户外投放对这类品牌广告主有什么作用？
- **must_preserve:** ["线下户外投放", "品牌广告主", "作用"]
- **must_not_add:** ["特定时间地点", "线上成交", "精准广触达", "营销可验证", "提升效率"]
- **expected_clarifying_question:** 您指的是哪种线下投放形式？
- **clarification_slots:** ["投放形式（如户外、店内、活动等）"]

---

## QH10-1 · 清晰问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q057`
- **source_document:** 全是变化的信号：2025私域趋势白皮书暨年度调研报告
- **user_query:** 在私域收入1千万到5千万的团队中，哪个行业的团队数量最多？
- **expected_route:** `clear`
- **canonical_query:** 在私域收入1千万到5千万的团队中，哪个行业的团队数量最多？
- **must_preserve:** ["私域收入", "1千万到5千万", "团队", "行业", "数量最多"]
- **must_not_add:** ["鞋服箱包", "见实科技", "具体年份", "其他行业名称"]

---

## QH10-2 · 可扩写问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q057`
- **source_document:** 全是变化的信号：2025私域趋势白皮书暨年度调研报告
- **user_query:** 私域收入在一千万到五千万之间的团队，哪个行业的团队数量排第一？
- **expected_route:** `expandable`
- **canonical_query:** 在私域收入1千万到5千万的团队中，哪个行业的团队数量最多？
- **must_preserve:** ["私域收入", "1千万到5千万", "团队", "行业", "数量最多"]
- **must_not_add:** ["鞋服箱包", "见实科技", "具体年份", "其他行业名称"]

---

## QH10-3 · 模糊问题

- **review_decision:** `yes`
- **review_notes:** ``
- **source_evaluation_id:** `Q057`
- **source_document:** 全是变化的信号：2025私域趋势白皮书暨年度调研报告
- **user_query:** 哪个行业的团队数量最多？
- **expected_route:** `ambiguous`
- **canonical_query:** 在私域收入1千万到5千万的团队中，哪个行业的团队数量最多？
- **must_preserve:** ["私域收入", "1千万到5千万", "团队", "行业", "数量最多"]
- **must_not_add:** ["鞋服箱包", "见实科技", "具体年份", "其他行业名称"]
- **expected_clarifying_question:** 您指的是哪个收入范围或场景下的团队数量？
- **clarification_slots:** ["收入范围", "团队所属领域或场景"]

---
