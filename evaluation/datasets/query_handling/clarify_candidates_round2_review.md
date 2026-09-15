# Clarify Candidates Round 2 Review

这一批专门测试没有前文时的指代缺失。

- 真正无法确定指代对象：填写 `yes`。
- 即使没有前文仍可直接回答：填写 `no`。
- 可直接修改问题、缺失信息和参考反问。

## QHD01

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 这种新的投资形式，其核心是什么？
- **canonical_query:** 种草作为一种新的投资形式，其核心是什么？
- **source_evaluation_id:** `Q003`
- **source_document:** 2024年Social&KOL营销趋势报告
- **standard_answer:** 品牌心智绑定（或称之为关键词绑定）
- **missing_required_information:** ["‘这种新的投资形式’所指的具体形式"]
- **expected_clarifying_question:** "您所说的‘这种新的投资形式’具体指什么？"
- **ambiguity_explanation:** 可能指种草、品牌广告、效果广告等不同形式，核心概念会不同。

---

## QHD02

- **review_decision:** `no`
- **review_notes:** ``
- **user_query:** 这一年小红书达人商单的CPE呈现什么趋势？
- **canonical_query:** 2023年小红书达人商单的CPE在全年呈现什么趋势？
- **source_evaluation_id:** `Q005`
- **source_document:** 2024年Social&KOL营销趋势报告
- **standard_answer:** 小红书整体流量竞争较为激烈，全年CPE呈显著上升趋势。
- **missing_required_information:** ["‘这一年’所指的年份"]
- **expected_clarifying_question:** "您想了解哪一年的小红书达人商单CPE趋势？"
- **ambiguity_explanation:** 不同年份的CPE趋势可能不同，‘这一年’缺少对话中的时间指代。

---

## QHD03

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 这两个大促期间的CPE表现有什么不同？
- **canonical_query:** 2023年小红书达人商单在618大促和双十一期间的CPE表现有何不同？
- **source_evaluation_id:** `Q006`
- **source_document:** 2024年Social&KOL营销趋势报告
- **standard_answer:** 618大促期间的CPE相对双十一表现更优。
- **missing_required_information:** ["需要比较的两个大促", "平台或商单范围"]
- **expected_clarifying_question:** "您想比较哪两个大促，以及哪个平台的CPE表现？"
- **ambiguity_explanation:** 618、双十一、年货节等组合及不同平台会产生不同答案。

---

## QHD04

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 这个品类在哪个月份的达人商单投放费用达到峰值？
- **canonical_query:** 2024年3C产品在哪个月份的品牌达人商单投放费用达到当年峰值？
- **source_evaluation_id:** `Q007`
- **source_document:** 2024年Social&KOL营销趋势报告
- **standard_answer:** 9月
- **missing_required_information:** ["‘这个品类’所指的产品品类", "年份"]
- **expected_clarifying_question:** "您指的是哪个产品品类、哪一年的达人商单投放？"
- **ambiguity_explanation:** 3C、美妆、母婴等品类以及不同年份的费用峰值月份可能不同。

---

## QHD05

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 这两家公司合并后获得了哪些投资，投资金额和估值是多少？
- **canonical_query:** 2023年底零食很忙与赵一鸣零食合并后，获得了哪些产业资本的投资？投资金额和估值分别是多少？
- **source_evaluation_id:** `Q015`
- **source_document:** 2025中国量贩零食行业现状报告
- **standard_answer:** 合并后获得了盐津铺子、好想你等产业资本超10亿元投资，估值突破105亿元。
- **missing_required_information:** ["‘这两家公司’所指的公司名称"]
- **expected_clarifying_question:** "您所说的两家公司分别是哪两家？"
- **ambiguity_explanation:** 不同公司合并事件对应的投资方、金额和估值完全不同。

---

## QHD06

- **review_decision:** `no`
- **review_notes:** ``
- **user_query:** 截至当时，门店数量排名前五的品牌有哪些？
- **canonical_query:** 2024年11月，国内门店数量排名前五的量贩零食品牌分别有哪些？
- **source_evaluation_id:** `Q022`
- **source_document:** 2025中国量贩零食行业现状报告
- **standard_answer:** 赵一鸣（近8000家）、零食很忙（近7000家）、好想来（近6000家）、零食有鸣（超过3000家）和爱零食（约3000家）。
- **missing_required_information:** ["‘当时’所指的时间", "所属行业"]
- **expected_clarifying_question:** "您想了解哪个时间点、哪个行业的门店数量排名？"
- **ambiguity_explanation:** 时间和行业不同，门店数量前五品牌会发生变化。

---

## QHD07

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 这两个年龄群体合计占宠物市场多少份额？
- **canonical_query:** 2024年，90后和00后合计占宠物市场的份额是多少？
- **source_evaluation_id:** `Q027`
- **source_document:** 2025年中国宠物行业市场报告
- **standard_answer:** 67.7%
- **missing_required_information:** ["‘这两个年龄群体’所指的群体", "年份"]
- **expected_clarifying_question:** "您指的是哪两个年龄群体？"
- **ambiguity_explanation:** 90后与00后、80后与90后等组合会得到不同占比，年份也会影响结果。

---

## QHD08

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 一线城市中，采用这种工具的宠物医院占比是多少？
- **canonical_query:** 一线城市中引入AI辅助诊断工具的宠物医院占比是多少？
- **source_evaluation_id:** `Q033`
- **source_document:** 2025年中国宠物行业市场报告
- **standard_answer:** 超过七成。
- **missing_required_information:** ["‘这种工具’所指的工具"]
- **expected_clarifying_question:** "您所说的‘这种工具’具体指哪一种工具？"
- **ambiguity_explanation:** AI辅助诊断、线上问诊或其他医疗工具的采用比例并不相同。

---

## QHD09

- **review_decision:** `no`
- **review_notes:** ``
- **user_query:** 为什么这些行业的品牌广告主会精细调整整体投放策略？
- **canonical_query:** 为什么同质化竞争激烈且数字化程度高的行业，品牌广告主会精细调整整体投放策略？
- **source_evaluation_id:** `Q043`
- **source_document:** 2025年品牌营销趋势报告
- **standard_answer:** 因为这类行业竞争激烈，或受政策推动、市场关注，亦或是日常刚需、陪伴类行业，产品受众广泛或呈多元化增长，单一媒体渠道无法满足其复杂营销需求。出于成本考量，品牌广告主会精细调整整体投放策略，在不同媒介点位中寻找最佳平衡，实现精准广覆盖。
- **missing_required_information:** ["‘这些行业’所指的行业类型"]
- **expected_clarifying_question:** "您所说的‘这些行业’具体指哪些行业？"
- **ambiguity_explanation:** 同质化竞争行业、低数字化行业或其他行业调整策略的原因可能不同。

---

## QHD10

- **review_decision:** `no`
- **review_notes:** ``
- **user_query:** 这类广告主对数字广告和户外广告预算的态度如何？
- **canonical_query:** 探索型广告主对数字和户外广告预算的态度如何？
- **source_evaluation_id:** `Q052`
- **source_document:** 2025年品牌营销趋势报告
- **standard_answer:** 探索型广告主对数字和户外广告预算均有较为乐观的态度，分别有45%和46%的广告主认为这两方面的营销预算会有所增加。
- **missing_required_information:** ["‘这类广告主’所指的广告主类型"]
- **expected_clarifying_question:** "您所说的‘这类广告主’具体是哪一类广告主？"
- **ambiguity_explanation:** 探索型、过程型和结果型广告主的预算态度不同。

---
