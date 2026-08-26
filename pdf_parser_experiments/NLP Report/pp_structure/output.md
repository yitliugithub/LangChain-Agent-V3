# PP-Structure Output

Source: NLP Report.pdf

Mode: full

DPI scale: 2.0

Page range: 1-3



## Page 1


### Result 1: markdown

# Climate-Science Claim Verification: A Multi-Stage Cascade System with  Adaptive Evidence Selection 

Group 75

## Abstract 

This project investigates the task of climatescience claim verification, specifically by enabling the system to retrieve relevant evidence from a large-scale evidence corpus given a claim, and to predict the claim's veracity label based on the evidence. Given that the system's final output includes both claim labels and evidence ids, this task requires the model to possess the ability to both retrieve evidence and verify claims, rather than merely performing claim-level classification. We build a multistage cascade system, first combining lexical retrieval and dense retrieval to generate candidate evidence, and then improving the ranking quality of the candidate evidence through hybrid fusion and cross-encoder reranking. Then, the system uses the top-ranked evidence to predict the claim label with a BERT-based classifier and generates the final evidence list through adaptive evidence selection. The goal of this design is to reduce the computational pressure brought by large-scale evidence retrieval,the impact of claim-evidence expression differences, and pipeline error propagation, while achieving a better balance between evidence quality and label accuracy. On the development set, the final system achieved an evidence retrieval F-score of 0.2042, a claim classification accuracy of 0.5130, and a harmonic mean of 0.2921.


## 1 Introduction 

This project focuses on automated fact-checking in the field of climate science. In this task, the veracity of the claim cannot be determined solely based on the claim text itself, but needs to be verified in conjunction with the retrieved evidence. We formulate this task as an interdependent process of evidence retrieval and claim verification: the former determines what evidence the system can access, while the latter determines whether the system can make a correct semantic judgment based on that evidence. In other words, the system needs both the ability to retrieve evidence from a largescale corpus and also the ability to make semantic reasons based on that evidence.



Based on this task structure, the key to system design is not merely selecting a classifier, but handling the dependencies among evidence retrieval,evidence ranking, and label prediction. First, the task is constrained by the scale of the evidence corpus. According to the notebook's statistics, the evidence corpus contains over 1.2 million evidence passages. Therefore, the system cannot directly apply deep neural matching between each claim and all evidence passages, and must first reduce the candidate space. At the same time, climatescience text itself also increases the difficulty of retrieval. Relevant claims may contain domainspecific terms, and some evidence passages may use different but semantically related expressions.Therefore, the system needs to consider the effects of both lexical matching and semantic matching.

In addition to candidate evidence recall, the system also needs to account for error propagation in the pipeline. Our final classifier does not make predictions independently of retrieval results, but rather constructs the claim-evidence context based on the top-ranked evidence after reranking. If key evidence does not enter the candidate pool, it is difficult for the downstream reranker and classifier to recover from this error. Therefore, early retrieval recall will directly affect the final label prediction.Meanwhile, since evaluation measures both evidence quality and label accuracy, the system also needs to strike a balance between evidence precision, evidence recall, and label prediction. These factors collectively indicate that the task requires a hierarchical system capable of simultaneously handling candidate evidence retrieval, evidence reranking, label prediction, and evidence selection.


## Page 2


### Result 1: markdown

## 2 Task and Evaluation 

In this task, the system needs to generate two inteerrelated predictions from the claim files and the evidence corpus: on the one hand, it needs to predict a veracity label for each claim; on the other hand, it needs to return evidence ids that support the judg ment. Therefore, the input-output format of the task inherently connects claim verification with evidence retrieval. Specifically, each claim contains a claim id and claim text, while the evidence corpus provides evidence passages that can be retrieved by the system. In the training set and development set, each claim also includes a gold claim label and gold evidence ids, which can be used for model training and development-set evaluation. However,the system needs to generate predictions for each claim's label and corresponding evidence because the test set only contains claim information.

The output of the claim label task is given four label spaces, including SUPPORTS,REFUTES, NOT_ENOUGH_INFO, and DISPUTED. SUPPORTS and REFUTES indicate that the evidence supports or refutes the claim.NOT_ENOUGH_INFO indicates that the existing evidence is insufficient to determine the veracity.DISPUTED indicates that the evidence contains disagreement or conflicting conclusions. Unlike the true/false binary classification, this label space is more complex because the system not only needs to identify clear cases of support or refutation but also tell the difference between "insufficient evidence" and "disputed evidence".



According to the reading results from the notebook, the project data includes 1,228 training claims, 154 development claims, 153 unlabelled test claims, and 1,208,827 evidence passages. This means that the number of claims is relatively limited, but the evidence corpus is very large, creating a highly asymmetric retrieval space. Therefore, subsequent models cannot directly process the complete evidence corpus but rather first narrow the search range to a manageable candidate set through evidence retrieval. The training label distribution is also imbalanced, with SUPPORTS, NOT_ENOUGH_INFO, REFUTES and DISPUTED appearing 519, 386, 199 and 124 times respectively. This suggests that the model might trade-off its ability to predict less frequent classes such as REFUTES and DISPUTED, if it overencodes majority-class patterns.



The evaluation is structured as dual-output for 

the task. Evidence retrieval F-score (F) measures the match between predicted evidence ids and gold evidence ids, while claim classification accuracy (A) measures whether the predicted claim label is correct. The two are combined using the harmonic mean (H):

2·F ·A 

<div style="text-align: center;"><img src="imgs/img_in_formula_box_782_282_928_335.jpg" alt="Image" width="12%" /></div>


This design makes the final score susceptible to imbalance in performance of the two subtasks. Therefore, the evaluation encourages the system to reach a stable trade-off between the quality of evidence and label prediction.



## 3 Data Analysis and Preprocessing 

Before formal modelling, we first read, checked,and preprocessed the project data based on the structure of the claim files and evidence.json. Since the system needs to perform evidence retrieval and claim classification in the later stages, we mainly checked whether the data can be structured in a way that is suitable for retrieval and classification,rather than treating each claim as a stand-alone text input to the model.



Through the data loading results, we confirmed that the main computational pressure comes from the scale of the evidence corpus. Although the number of claims is relatively limited, the corpus contains over 1.2 million evidence passages, so the later models cannot directly scan the entire corpus for deep matching. We thus first converted the evidence corpus into searchable data representations,to prepare it for later candidate evidence retrieval.

We also checked the label distribution and gold evidence distribution of the training set. The label distribution shows that the minority-class samples are fewer than the majority class, which suggests that the subsequent classifier may be affected by class imbalance. In addition, each claim usually corresponds to multiple gold evidences. The average number of evidence for the training set is 3.36 and the average number of evidence for the development set is 3.19. Therefore, evidence output cannot simply be regarded as a single evidence selection problem, but needs to balance between retaining suffi cient evidence and reducing irrelevant evidence.



In the preprocessing step, we mainly converted the raw evidence corpus into formats that could be used for later retrieval. The code generated tokenized evidence for BM25 and used sentencetransformers/all-MiniLM-L6-v2 to generate dense 


## Page 3


### Result 1: markdown

embeddings. Due to the high constructing cost, we used a caching mechanism to store intermediate outputs such as the sparse index, TF-IDF matrix,and dense embeddings. Later experiments could reuse these intermediate results without processing the entire evidence corpus from scratch each time.

Through data preprocessing, the raw data is organised into formats that could be directly used in later stages. The Method section builds on this prepared data and details the design of candidate retrieval, reranking, classification, and related components.



<div style="text-align: center;"><img src="imgs/img_in_image_box_109_485_555_1040.jpg" alt="Image" width="37%" /></div>


<div style="text-align: center;">Figure 1: Final architecture of the proposed factverification pipeline. </div>


## 4 Methodology 

Our system follows a retrieval-centred cascade architecture rather than treating climate-science fact verification as a direct four-way text classification problem. This design is motivated by the output structure of the task: for each claim, the system must predict both a veracity label and a compact set of evidence identifiers. Since the evidence corpus contains more than one million passages, it is computationally infeasible to apply a deep neural reader to every possible claim–evidence pair. We therefore decompose the task into four connected stages: hybrid candidate retrieval, cross-encoder reranking, claim-evidence classification, and adaptive evidence selection. The first two stages determine what information is available to the reader,while the latter two stages determine the submitted label and evidence set. This pipeline structure also makes the system easier to analyse, because retrieval errors, ranking errors, and classification errors can be examined separately.



### 4.1 Hybrid Candidate Retrieval 

The first stage aims to construct a candidate evidence pool with high recall while keeping the number of passages small enough for neural reranking.A purely lexical retriever is often reliable when a climate claim contains explicit scientific entities,numerical values, dates, or technical terms. However, it can fail when the evidence expresses the same information using different wording. Conversely, a dense retriever can recover semantically related passages, but it may also rank topically similar yet non-evidential passages too highly. For this reason, our final implementation combines sparse lexical matching and dense semantic retrieval instead of relying on a single retrieval signal.

BM25 is used as the main sparse retriever because exact term matching is still important for scientific fact verification, especially for claims involving quantities, named reports, climate indicators, and domain-specific terms such as "CO2","temperature anomaly", and "sea level" (1). In the final configuration, we set k1 = 1.2 and b = 0.85,which is consistent with our experimental setup.The relatively high length-normalisation parameter is used to reduce the tendency of long evidence passages to receive high scores simply because they contain many query terms. To make the formula fit the ACL two-column layout, we write the denominator separately:

<div style="text-align: center;"><img src="imgs/img_in_formula_box_626_1209_1041_1315.jpg" alt="Image" width="34%" /></div>


To complement BM25, we add a TF-IDF retriever with unigram and bigram features. This branch is useful for semi-fixed scientific expressions such as "global warming", "sea level rise",or "greenhouse gas", where phrase-level overlap is more informative than isolated word overlap. In parallel, we use all-MiniLM-L6-v2 as a dense biencoder to embed claims and evidence passages into the same vector space (2; 3). Dense retrieval 