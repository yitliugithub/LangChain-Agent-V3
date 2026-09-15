# Additional Clarify Candidates Review

目标是为二分类路由补充真正需要反问的问题。

## 审核方法

- 如果缺失信息确实会产生多个明显不同的答案，将 `review_decision` 填为 `yes`。
- 如果问题虽然宽泛但仍可合理回答，填写 `no`。
- 可以直接修改问题、缺失信息和参考反问。
- `ambiguity_explanation` 只是帮助审核，不会提供给路由模型。

## QHC01

- **review_decision:** `no`
- **review_notes:** ``
- **user_query:** 投放抖音达人时，应该聚焦哪个层级的达人？
- **canonical_query:** 根据报告，在美妆行业投放抖音达人时，应聚焦哪个层级的达人？
- **source_evaluation_id:** `Q004`
- **source_document:** 2024年Social&KOL营销趋势报告
- **standard_answer:** 应聚焦T2-T3层级的达人。
- **missing_required_information:** ["行业"]
- **expected_clarifying_question:** "请问这是哪个行业的抖音达人投放？"
- **ambiguity_explanation:** 美妆、3C、母婴等行业的达人层级策略可能不同；缺少行业会得到不同答案。

---

## QHC02

- **review_decision:** `no`
- **review_notes:** ``
- **user_query:** 根据报告，2024年中国银发旅游市场规模大约是多少？
- **canonical_query:** 根据报告，2024年中国银发旅游市场规模大约是多少？预计到2028年将达到多少？
- **source_evaluation_id:** `Q041`
- **source_document:** 2025年中国银发经济研究报告
- **standard_answer:** 2024年市场规模约为1.6万亿人民币，预计到2028年将达到约2.7万亿人民币。
- **missing_required_information:** ["预测时间点"]
- **expected_clarifying_question:** "您希望了解哪一年的市场规模预测？例如，是2028年的预测，还是其他年份？"
- **ambiguity_explanation:** 该问题仅询问了2024年的市场规模，但未指定需要预测的未来年份。如果用户想要2028年的预测，答案约为2.7万亿；如果用户想要其他年份（如2025年或2030年），答案将不同。因此，缺少预测时间点会导致答案不唯一。

---

## QHC03

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 2024年实施的战略是什么，效果如何？
- **canonical_query:** 三只松鼠在2024年实施了什么战略，其效果如何？
- **source_evaluation_id:** `Q018`
- **source_document:** 2025中国量贩零食行业现状报告
- **standard_answer:** 三只松鼠在2024年实施了“高端性价比+全渠道”战略，通过并购“爱零食”获得1800家线下门店，分销网点突破万家；同时抖音渠道GMV增长180%，带动全年营收预计达102-108亿元，同比增加43%-52%。
- **missing_required_information:** ["品牌主体"]
- **expected_clarifying_question:** "您指的是哪个品牌或公司在2024年实施的战略？"
- **ambiguity_explanation:** 该问题未指明品牌主体，可能指三只松鼠、良品铺子、盐津铺子等不同公司，它们各自在2024年实施了不同的战略（如三只松鼠的'高端性价比+全渠道'，良品铺子的'降价不降质'等），且效果各异。因此无法确定具体战略和效果，需先明确品牌。

---

## QHC04

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 两种连锁扩张模式在资金压力上有什么不同？
- **canonical_query:** 直营连锁模式和加盟扩张模式在资金压力方面有什么不同？
- **source_evaluation_id:** `Q011`
- **source_document:** 2025中国量贩零食行业现状报告
- **standard_answer:** 直营连锁模式需要企业投入大量资金用于门店租赁、装修和日常运营，资金压力较大；加盟扩张模式则更注重轻资产运营，由加盟商负责门店的租赁、装修和日常运营，可以大幅降低企业的资金压力。
- **missing_required_information:** ["需要比较的两种扩张模式"]
- **expected_clarifying_question:** "您想比较哪两种扩张模式，例如直营连锁和加盟扩张吗？"
- **ambiguity_explanation:** 直营、加盟、合伙等模式的资金承担方式不同；未说明比较对象时无法确定答案。

---

## QHC05

- **review_decision:** `no`
- **review_notes:** ``
- **user_query:** 2024年宠物行业的销售情况怎么样？
- **canonical_query:** 2024年宠物类目全平台线上销售额是多少？同比增长了多少？
- **source_evaluation_id:** `Q029`
- **source_document:** 2025年中国宠物行业市场报告
- **standard_answer:** 2024年宠物类目全平台线上销售额为502.3亿元，同比增加10.0%。
- **missing_required_information:** ["统计指标", "统计范围"]
- **expected_clarifying_question:** "您想了解2024年宠物行业在哪个统计指标上的表现？例如，是全平台线上销售额，还是线上销售额的同比增长率？另外，统计范围是全平台线上，还是包括线下渠道？"
- **ambiguity_explanation:** 该问题未明确指定统计指标，可能指销售额（如502.3亿元），也可能指同比增长率（如10.0%）。同时，统计范围可能限定为线上全平台，也可能包含线下渠道，导致答案不同。因此，需要用户澄清具体指标和范围。

---

## QHC06

- **review_decision:** `no`
- **review_notes:** ``
- **user_query:** 养宠人群在宠物食品和宠物医疗方面的消费行为有何不同？
- **canonical_query:** 根据报告，三线及以下县域养宠人群在宠物食品和宠物医疗方面的消费行为有何不同？
- **source_evaluation_id:** `Q035`
- **source_document:** 2025年中国宠物行业市场报告
- **standard_answer:** 三线及以下县域养宠人群在宠物食品上更倾向于实惠的选择，如散装粮和自制猫饭，但在宠物医疗方面舍得投入。
- **missing_required_information:** ["城市层级或地区范围"]
- **expected_clarifying_question:** "您想了解哪个城市层级或地区的养宠人群？"
- **ambiguity_explanation:** 一线城市与三线及以下县域的宠物食品和医疗消费行为可能不同，缺少地区范围无法确定结论。

---

## QHC07

- **review_decision:** `no`
- **review_notes:** ``
- **user_query:** 为什么线上线下的协作能力对这类品牌广告主尤为重要？
- **canonical_query:** 为什么线上线下的协作能力对这类品牌广告主尤为重要？
- **source_evaluation_id:** `Q044`
- **source_document:** 2025年品牌营销趋势报告
- **standard_answer:** 因为线上可以尽快缩短消费链路，提升对数字媒体的依赖度，但线下户外投放能在特定时间地点触达对应人群，促成线上成交，所以线上线下的协作能力提升对其尤为重要。
- **missing_required_information:** ["‘这类品牌广告主’的具体类型"]
- **expected_clarifying_question:** "您所说的‘这类品牌广告主’具体指哪一类？"
- **ambiguity_explanation:** ‘这类’缺少前文指代，不同经营目标或行业的品牌广告主需要线上线下协作的原因可能不同。

---

## QHC08

- **review_decision:** `no`
- **review_notes:** ``
- **user_query:** 春节期间哪些品类的消费在县域地区表现活跃？
- **canonical_query:** 根据报告，2025年春节期间，哪些品类的消费在县域地区表现活跃？
- **source_evaluation_id:** `Q048`
- **source_document:** 2025年品牌营销趋势报告
- **standard_answer:** 县域地方菜、火锅、奶茶咖啡消费活跃，相关品类外卖、团购量同比实现两倍以上增长。
- **missing_required_information:** ["年份"]
- **expected_clarifying_question:** "您指的是哪一年的春节期间？"
- **ambiguity_explanation:** 该问题未指明具体年份，而不同年份的春节消费趋势可能不同。例如，2025年报告显示县域地方菜、火锅、奶茶咖啡消费活跃，但其他年份可能因经济环境、政策或流行趋势而出现不同活跃品类。因此，缺少年份信息会导致答案不确定，无法安全作答。

---

## QHC09

- **review_decision:** `yes`
- **review_notes:** ``
- **user_query:** 养宠人群对这两种购买渠道的偏好度分别是多少？
- **canonical_query:** 根据报告，养宠人群在购买宠物产品时，对大型综合电商平台和线上直播间的偏好度分别是多少？
- **source_evaluation_id:** `Q030`
- **source_document:** 2025年中国宠物行业市场报告
- **standard_answer:** 养宠人群对大型综合电商平台的偏好度为68.1%，对线上直播间的偏好度为18.9%。
- **missing_required_information:** ["需要比较的两种购买渠道"]
- **expected_clarifying_question:** "您指的是大型综合电商平台和线上直播间吗？"
- **ambiguity_explanation:** ‘这两种渠道’指代不明，可能是综合电商与直播间，也可能是其他渠道组合。

---

## QHC10

- **review_decision:** `no`
- **review_notes:** ``
- **user_query:** 有多少广告主计划增加营销预算？
- **canonical_query:** 根据报告，有多少比例的广告主计划增加对中线及下沉市场的营销预算？
- **source_evaluation_id:** `Q049`
- **source_document:** 2025年品牌营销趋势报告
- **standard_answer:** 31%的广告主认为今年将会增加对这些市场的营销预算。
- **missing_required_information:** ["年份", "预算投向的市场范围"]
- **expected_clarifying_question:** "您想了解哪一年、针对哪类市场范围的增投计划？"
- **ambiguity_explanation:** 不同年份以及中线、下沉或整体市场的增投比例不同，当前问题无法确定唯一数字。

---
