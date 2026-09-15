# Binary Query Routing Review

本轮只判断：问题应该直接进入检索，还是必须先反问。

## 修改规则

- `retrieve`：当前信息足以形成一个合理答案，即使问题比较宽泛或口语化。
- `clarify`：缺少的信息会导致多个明显不同的答案，不能安全检索。
- 如果改成 `retrieve`，请将 `missing_required_information` 改为 `[]`，将 `expected_clarifying_question` 改为 `null`。
- 如果保留或改成 `clarify`，请检查缺失信息和参考反问是否真正必要。
- `review_decision` 最终填写 `yes`；认为问题本身不适合评测时填写 `no`。
- 模型预测只用于帮助发现争议，不是标准答案。

## QH01-1

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 根据报告，用户为什么开始喜欢去品牌自播环境？
- **previous_route:** `clear`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `ambiguous`
- **model_reason:** 问题中“报告”指代不明，且未说明具体品牌或行业，存在多种合理解释，需先确认报告范围。

---

## QH01-2

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 报告里说，用户为啥现在更爱去品牌自己的直播间了？
- **previous_route:** `expandable`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `expandable`
- **model_reason:** 问题意图明确，但口语化且缺少标准检索术语，可改写为更规范的表达而不需猜测关键条件。

---

## QH01-3

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 用户为什么更喜欢某种直播方式？
- **previous_route:** `ambiguous`
- **expected_action:** `clarify`
- **missing_required_information:** ["直播类型（品牌自播或达人直播）"]
- **expected_clarifying_question:** "您指的是品牌自播还是达人直播？"
- **model_predicted_route:** `ambiguous`
- **model_reason:** 问题缺少具体对象（哪种直播方式）、比较对象（与何种方式对比）和指标（偏好如何衡量），存在多种合理解释，需先澄清。

---

## QH02-1

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 根据报告，在小红书被美妆护肤香水品类种草后，有多大比例的用户会点击链接查看详情？
- **previous_route:** `clear`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `clear`
- **model_reason:** 用户问题明确了对象（小红书美妆护肤香水品类）、行为（种草后点击链接查看详情）和指标（比例），可直接用于检索。

---

## QH02-2

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 报告里说，在小红书上看了美妆护肤香水的内容后，有多少人会点链接看详情？
- **previous_route:** `expandable`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `ambiguous`
- **model_reason:** 用户提到“报告里说”，但未指明具体报告，且“多少人”缺少明确的时间范围或统计口径，存在多个合理解释。

---

## QH02-3

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 在小红书被种草后，用户会点击链接查看详情吗？
- **previous_route:** `ambiguous`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `clear`
- **model_reason:** 问题对象（小红书种草用户）、行为（点击链接查看详情）和任务（了解行为是否发生）明确，可直接检索。

---

## QH03-1

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 根据报告，2011年至今酒类市场在味型上有哪些创新趋势？
- **previous_route:** `clear`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `clear`
- **model_reason:** 问题明确了对象（酒类市场）、时间范围（2011年至今）和任务（味型创新趋势），可直接用于检索。

---

## QH03-2

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 报告里说从2011年到现在，酒类口味方面有哪些新的发展方向？
- **previous_route:** `expandable`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `ambiguous`
- **model_reason:** 用户提到“报告里说”，但未指明具体报告，且“酒类口味”范围过宽，无法确定检索对象。

---

## QH03-3

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 酒类市场有哪些创新趋势？
- **previous_route:** `ambiguous`
- **expected_action:** `retrieve`
- **missing_required_information:** ["时间范围", "创新趋势的具体方面（如味型、包装、营销等）"]
- **expected_clarifying_question:** "您指的是哪个时间范围？以及您关注的是味型还是其他方面的创新趋势？"
- **model_predicted_route:** `expandable`
- **model_reason:** 问题意图明确，但“创新趋势”表述较笼统，可改写为更适合检索的标准表达。

---

## QH04-1

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 万辰集团的门店数量从2022年到2024年7月增长了多少家？
- **previous_route:** `clear`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `clear`
- **model_reason:** 问题明确了对象（万辰集团）、时间范围（2022年至2024年7月）和指标（门店数量增长），可直接检索。

---

## QH04-2

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 万辰集团的门店数量在2022年到2024年7月之间增加了多少？
- **previous_route:** `expandable`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `clear`
- **model_reason:** 问题明确了对象（万辰集团）、时间范围（2022年至2024年7月）和任务（门店数量增加量），可直接用于检索。

---

## QH04-3

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 万辰集团的门店数量增长了多少？
- **previous_route:** `ambiguous`
- **expected_action:** `retrieve`
- **missing_required_information:** ["起始时间", "结束时间"]
- **expected_clarifying_question:** "您指的是哪个时间段内的门店数量增长？"
- **model_predicted_route:** `clear`
- **model_reason:** 问题对象（万辰集团）、指标（门店数量增长）和任务（查询增长量）明确，可直接检索。

---

## QH05-1

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 为什么直营连锁模式扩张速度相对较慢？
- **previous_route:** `clear`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `clear`
- **model_reason:** 问题对象（直营连锁模式）、任务（解释扩张速度慢的原因）明确，可直接用于知识库检索。

---

## QH05-2

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 直营连锁为什么扩张得慢？
- **previous_route:** `expandable`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `expandable`
- **model_reason:** 问题意图明确，但表述口语化，可改写为更利于检索的标准表达。

---

## QH05-3

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 为什么扩张速度相对较慢？
- **previous_route:** `ambiguous`
- **expected_action:** `clarify`
- **missing_required_information:** ["扩张模式"]
- **expected_clarifying_question:** "您指的是什么对象的扩张模式？"
- **model_predicted_route:** `ambiguous`
- **model_reason:** 问题缺少对象（谁在扩张）和比较基准（相对什么较慢），存在多个合理解释，需先澄清。

---

## QH06-1

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 2024年养宠人在选择宠物主粮品牌时，最看重的因素是什么？
- **previous_route:** `clear`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `clear`
- **model_reason:** 问题明确了对象（2024年养宠人）、范围（宠物主粮品牌选择）和任务（最看重的因素），可直接用于检索。

---

## QH06-2

- **review_decision:** `yes
- **review_notes:** ``
- **user_query:** 2024年大家挑宠物主粮牌子的时候，最在意啥？
- **previous_route:** `expandable`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `expandable`
- **model_reason:** 问题意图明确，但口语化，可改写为更利于检索的标准表达。

---

## QH07-1

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 与国际知名品牌相比，中国宠物行业在品牌建设和执行标准上存在哪些不足？
- **previous_route:** `clear`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `clear`
- **model_reason:** 问题对象（中国宠物行业与国际知名品牌）、比较维度（品牌建设和执行标准）和任务（指出不足）均明确，可直接检索。

---

## QH07-2

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 和国际大牌比，中国宠物行业差在哪儿？
- **previous_route:** `expandable`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `expandable`
- **model_reason:** 问题意图明确，但表述口语化，可改写为更利于检索的标准表达。

---

## QH07-3

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 中国宠物行业存在哪些不足？
- **previous_route:** `ambiguous`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `expandable`
- **model_reason:** 问题对象（中国宠物行业）和任务（找不足）明确，但表达较笼统，可改写为更适合检索的标准表述。

---

## QH08-1

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 根据工信部数据，截至2023年，我国养老用品市场规模达到多少？
- **previous_route:** `clear`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `clear`
- **model_reason:** 问题包含明确的数据来源（工信部）、时间（2023年）和对象（养老用品市场规模），任务清晰，可直接检索。

---

## QH08-2

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 2023年我国养老用品市场规模有多大？
- **previous_route:** `expandable`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `clear`
- **model_reason:** 问题包含明确年份、对象（养老用品市场）和任务（查询规模），可直接用于检索。

---

## QH08-3

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 我国养老用品市场规模有多大？
- **previous_route:** `ambiguous`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `expandable`
- **model_reason:** 问题意图明确，但“养老用品市场”可标准化为更利于检索的行业术语，且无需新增条件。

---

## QH09-1

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 线下户外投放对品牌广告主有什么作用？
- **previous_route:** `clear`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `expandable`
- **model_reason:** 问题意图明确，但“线下户外投放”可标准化为“户外广告投放”以改善检索效果。

---

## QH09-2

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 品牌广告主做线下户外广告能带来哪些好处？
- **previous_route:** `expandable`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `expandable`
- **model_reason:** 问题意图明确，但'线下户外广告'可标准化为'户外广告'，且'好处'可改写为'价值'以提升检索效果。

---

## QH09-3

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 线下投放有什么作用？
- **previous_route:** `ambiguous`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `expandable`
- **model_reason:** 问题意图明确，但“线下投放”口语化，可改写为更标准的检索表达。

---

## QH10-1

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 在私域收入1千万到5千万的团队中，哪个行业的团队数量最多？
- **previous_route:** `clear`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `clear`
- **model_reason:** 问题明确了收入区间、统计对象和指标，可直接用于检索。

---

## QH10-2

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 私域收入在一千万到五千万之间的团队，哪个行业的团队数量排第一？
- **previous_route:** `expandable`
- **expected_action:** `retrieve`
- **missing_required_information:** []
- **expected_clarifying_question:** null
- **model_predicted_route:** `clear`
- **model_reason:** 问题明确了对象（私域收入一千万到五千万的团队）、范围（行业）和任务（数量排名第一），可直接检索。

---

## QH10-3

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 哪个行业的团队数量最多？
- **previous_route:** `ambiguous`
- **expected_action:** `clarify`
- **missing_required_information:** ["收入范围", "团队所属领域或场景"]
- **expected_clarifying_question:** "您指的是哪个场景下的团队数量？"
- **model_predicted_route:** `ambiguous`
- **model_reason:** 问题缺少必要的时间范围、数据来源或行业分类标准，不同解释会得到不同答案。

---
