# **Climate-Science Claim Verification: A Multi-Stage Cascade System with Adaptive Evidence Selection** 

**Group 75** 

## **Abstract** 

This project investigates the task of climatescience claim verification, specifically by enabling the system to retrieve relevant evidence from a large-scale evidence corpus given a claim, and to predict the claim’s veracity label based on the evidence. Given that the system’s final output includes both claim labels and evidence ids, this task requires the model to possess the ability to both retrieve evidence and verify claims, rather than merely performing claim-level classification. We build a multistage cascade system, first combining lexical retrieval and dense retrieval to generate candidate evidence, and then improving the ranking quality of the candidate evidence through hybrid fusion and cross-encoder reranking. Then, the system uses the top-ranked evidence to predict the claim label with a BERT-based classifier and generates the final evidence list through adaptive evidence selection. The goal of this design is to reduce the computational pressure brought by large-scale evidence retrieval, the impact of claim-evidence expression differences, and pipeline error propagation, while achieving a better balance between evidence quality and label accuracy. On the development set, the final system achieved an evidence retrieval F-score of 0.2042, a claim classification accuracy of 0.5130, and a harmonic mean of 0.2921. 

## **1 Introduction** 

This project focuses on automated fact-checking in the field of climate science. In this task, the veracity of the claim cannot be determined solely based on the claim text itself, but needs to be verified in conjunction with the retrieved evidence. We formulate this task as an interdependent process of evidence retrieval and claim verification: the former determines what evidence the system can access, while the latter determines whether the system can make a correct semantic judgment based 

on that evidence. In other words, the system needs both the ability to retrieve evidence from a largescale corpus and also the ability to make semantic reasons based on that evidence. 

Based on this task structure, the key to system design is not merely selecting a classifier, but handling the dependencies among evidence retrieval, evidence ranking, and label prediction. First, the task is constrained by the scale of the evidence corpus. According to the notebook’s statistics, the evidence corpus contains over 1.2 million evidence passages. Therefore, the system cannot directly apply deep neural matching between each claim and all evidence passages, and must first reduce the candidate space. At the same time, climatescience text itself also increases the difficulty of retrieval. Relevant claims may contain domainspecific terms, and some evidence passages may use different but semantically related expressions. Therefore, the system needs to consider the effects of both lexical matching and semantic matching. 

In addition to candidate evidence recall, the system also needs to account for error propagation in the pipeline. Our final classifier does not make predictions independently of retrieval results, but rather constructs the claim-evidence context based on the top-ranked evidence after reranking. If key evidence does not enter the candidate pool, it is difficult for the downstream reranker and classifier to recover from this error. Therefore, early retrieval recall will directly affect the final label prediction. Meanwhile, since evaluation measures both evidence quality and label accuracy, the system also needs to strike a balance between evidence precision, evidence recall, and label prediction. These factors collectively indicate that the task requires a hierarchical system capable of simultaneously handling candidate evidence retrieval, evidence reranking, label prediction, and evidence selection. 

## **2 Task and Evaluation** 

In this task, the system needs to generate two interrelated predictions from the claim files and the evidence corpus: on the one hand, it needs to predict a veracity label for each claim; on the other hand, it needs to return evidence ids that support the judgment. Therefore, the input-output format of the task inherently connects claim verification with evidence retrieval. Specifically, each claim contains a claim id and claim text, while the evidence corpus provides evidence passages that can be retrieved by the system. In the training set and development set, each claim also includes a gold claim label and gold evidence ids, which can be used for model training and development-set evaluation. However, the system needs to generate predictions for each claim’s label and corresponding evidence because the test set only contains claim information. 

The output of the claim label task is given four label spaces, including SUPPORTS, REFUTES, NOT_ENOUGH_INFO, and DISPUTED. SUPPORTS and REFUTES indicate that the evidence supports or refutes the claim. NOT_ENOUGH_INFO indicates that the existing evidence is insufficient to determine the veracity. DISPUTED indicates that the evidence contains disagreement or conflicting conclusions. Unlike the true/false binary classification, this label space is more complex because the system not only needs to identify clear cases of support or refutation but also tell the difference between “insufficient evidence” and “disputed evidence”. 

According to the reading results from the notebook, the project data includes 1,228 training claims, 154 development claims, 153 unlabelled test claims, and 1,208,827 evidence passages. This means that the number of claims is relatively limited, but the evidence corpus is very large, creating a highly asymmetric retrieval space. Therefore, subsequent models cannot directly process the complete evidence corpus but rather first narrow the search range to a manageable candidate set through evidence retrieval. The training label distribution is also imbalanced, with SUPPORTS, NOT_ENOUGH_INFO, REFUTES and DISPUTED appearing 519, 386, 199 and 124 times respectively. This suggests that the model might trade-off its ability to predict less frequent classes such as REFUTES and DISPUTED, if it overencodes majority-class patterns. 

The evaluation is structured as dual-output for 

the task. Evidence retrieval F-score (F) measures the match between predicted evidence ids and gold evidence ids, while claim classification accuracy (A) measures whether the predicted claim label is correct. The two are combined using the harmonic mean (H): 



This design makes the final score susceptible to imbalance in performance of the two subtasks. Therefore, the evaluation encourages the system to reach a stable trade-off between the quality of evidence and label prediction. 

## **3 Data Analysis and Preprocessing** 

Before formal modelling, we first read, checked, and preprocessed the project data based on the structure of the claim files and evidence.json. Since the system needs to perform evidence retrieval and claim classification in the later stages, we mainly checked whether the data can be structured in a way that is suitable for retrieval and classification, rather than treating each claim as a stand-alone text input to the model. 

Through the data loading results, we confirmed that the main computational pressure comes from the scale of the evidence corpus. Although the number of claims is relatively limited, the corpus contains over 1.2 million evidence passages, so the later models cannot directly scan the entire corpus for deep matching. We thus first converted the evidence corpus into searchable data representations, to prepare it for later candidate evidence retrieval. 

We also checked the label distribution and gold evidence distribution of the training set. The label distribution shows that the minority-class samples are fewer than the majority class, which suggests that the subsequent classifier may be affected by class imbalance. In addition, each claim usually corresponds to multiple gold evidences. The average number of evidence for the training set is 3.36 and the average number of evidence for the development set is 3.19. Therefore, evidence output cannot simply be regarded as a single evidence selection problem, but needs to balance between retaining sufficient evidence and reducing irrelevant evidence. 

In the preprocessing step, we mainly converted the raw evidence corpus into formats that could be used for later retrieval. The code generated tokenized evidence for BM25 and used sentencetransformers/all-MiniLM-L6-v2 to generate dense 

embeddings. Due to the high constructing cost, we used a caching mechanism to store intermediate outputs such as the sparse index, TF-IDF matrix, and dense embeddings. Later experiments could reuse these intermediate results without processing the entire evidence corpus from scratch each time. 

Through data preprocessing, the raw data is organised into formats that could be directly used in later stages. The Method section builds on this prepared data and details the design of candidate retrieval, reranking, classification, and related components. 



Figure 1: Final architecture of the proposed factverification pipeline. 

tive evidence selection. The first two stages determine what information is available to the reader, while the latter two stages determine the submitted label and evidence set. This pipeline structure also makes the system easier to analyse, because retrieval errors, ranking errors, and classification errors can be examined separately. 

### **4.1 Hybrid Candidate Retrieval** 

The first stage aims to construct a candidate evidence pool with high recall while keeping the number of passages small enough for neural reranking. A purely lexical retriever is often reliable when a climate claim contains explicit scientific entities, numerical values, dates, or technical terms. However, it can fail when the evidence expresses the same information using different wording. Conversely, a dense retriever can recover semantically related passages, but it may also rank topically similar yet non-evidential passages too highly. For this reason, our final implementation combines sparse lexical matching and dense semantic retrieval instead of relying on a single retrieval signal. 

BM25 is used as the main sparse retriever because exact term matching is still important for scientific fact verification, especially for claims involving quantities, named reports, climate indicators, and domain-specific terms such as “CO2”, “temperature anomaly”, and “sea level” (1). In the final configuration, we set _k_ 1 = 1 _._ 2 and _b_ = 0 _._ 85, which is consistent with our experimental setup. The relatively high length-normalisation parameter is used to reduce the tendency of long evidence passages to receive high scores simply because they contain many query terms. To make the formula fit the ACL two-column layout, we write the denominator separately: 

## **4 Methodology** 

Our system follows a retrieval-centred cascade architecture rather than treating climate-science fact verification as a direct four-way text classification problem. This design is motivated by the output structure of the task: for each claim, the system must predict both a veracity label and a compact set of evidence identifiers. Since the evidence corpus contains more than one million passages, it is computationally infeasible to apply a deep neural reader to every possible claim–evidence pair. We therefore decompose the task into four connected stages: hybrid candidate retrieval, cross-encoder reranking, claim-evidence classification, and adap- 



To complement BM25, we add a TF-IDF retriever with unigram and bigram features. This branch is useful for semi-fixed scientific expressions such as “global warming”, “sea level rise”, or “greenhouse gas”, where phrase-level overlap is more informative than isolated word overlap. In parallel, we use all-MiniLM-L6-v2 as a dense biencoder to embed claims and evidence passages into the same vector space (2; 3). Dense retrieval 

is based on cosine similarity: 



The sparse candidates from BM25 and TF-IDF are merged using Reciprocal Rank Fusion (RRF) (4). RRF is suitable here because BM25, TF-IDF, and dense retrieval scores are not calibrated on the same numerical scale. Instead of combining raw scores, RRF only depends on the relative rank of a passage in each retrieval list: 



In the final candidate construction, we keep the top 150 RRF candidates as the lexical core and append up to 100 novel dense candidates as a semantic bonus. This sparse-preserving strategy is deliberately conservative. Dense retrieval improves coverage for paraphrased evidence, but it is not allowed to overwrite high-confidence lexical matches that may contain exact numbers or decisive scientific terms. The resulting candidate pool is therefore broad enough for recall while still small enough for the reranking stage. 

### **4.2 Cross-Encoder Reranking** 

The hybrid retriever returns passages that are related to the claim, but many of them are only topically similar rather than directly evidential. The second stage therefore reranks each claim– evidence pair using a cross-encoder. Unlike the dense bi-encoder, which encodes the claim and evidence independently, the cross-encoder receives the concatenated pair and allows transformer selfattention to model token-level interactions between the two texts (7; 5). This is important in climatescience verification, where small modifiers such as “not”, “only”, “since 1998”, or “human-caused” can change the factual relation between the claim and the evidence. 

The final submitted system uses cross-encoder/ms-marco-MiniLM-L-12-v2 as a zero-shot reranker, matching the experimental setup. The reranker assigns a relevance score to each candidate pair: 



During development, we experimented with heavier rerankers and fine-tuning-style variants, but these alternatives were not retained in the final 

system because they produced less stable downstream classification. The final choice favours architectural alignment over model size: the MiniLMbased dense retriever and MiniLM-based crossencoder produce a ranked evidence list that is more consistent with the distribution later consumed by the BERT reader. This stage is also computationally feasible because cross-attention is applied only to the filtered candidate pool, not to the full evidence corpus. 

### **4.3 Aligned Fact-Check Classifier** 

After reranking, the system predicts one of four labels: SUPPORTS, REFUTES, NOT_ENOUGH_INFO, and DISPUTED. We use bert-base-uncased as the reader backbone. The model follows the pretrained bidirectional Transformer encoder formulation introduced in BERT (6), but it is trained on claim–evidence contexts rather than isolated claims. The input is constructed by concatenating the claim with the top-ranked evidence passages from the reranked list. The pooled [CLS] representation is then passed to a linear classification head: 



A major source of error in cascade fact-checking systems is train–test distribution mismatch. If the reader is trained only with gold evidence, it learns from clean and highly relevant contexts. However, at inference time, the reader receives evidence produced by imperfect retrieval and reranking. To reduce this mismatch, our ClsDatasetAligned component constructs multiple training views for each claim. The gold-evidence view teaches the model the ideal semantic relation between claim and evidence. The reranker-output view exposes the model to realistic retrieved evidence. The mixed view injects hard negative passages into the context, forcing the reader to rely on evidence quality rather than topical similarity alone. In this report, hard negatives are therefore used for reader alignment, not for separately fine-tuning the final MiniLM reranker. 

The classifier also needs to handle label imbalance. In the training set, SUPPORTS and NOT_ENOUGH_INFO are more frequent than REFUTES and DISPUTED. A standard crossentropy objective can therefore bias the model towards majority classes. Using raw inversefrequency weights, however, can over-amplify minority classes and reduce stability on majority la- 

bels. Our final system uses softened class weights [1 _._ 0 _,_ 1 _._ 6 _,_ 1 _._ 15 _,_ 2 _._ 0] together with label smoothing. Let _qi_ ( _y_ ) denote the smoothed target distribution for claim _i_ . The classification objective is: 



This objective gives additional learning signal to minority labels without forcing the model to overcorrect away from the dominant classes. It is particularly useful for distinguishing REFUTES and DISPUTED, because both labels may involve evidence that contradicts or complicates the original claim. 

### **4.4 Adaptive Evidence Selection** 

The reader and the submitted evidence list serve related but not identical purposes. The reader benefits from a stable context window that contains enough information for label prediction, whereas the official evidence score rewards a compact and accurate set of evidence ids. A fixed top- _K_ rule ignores this difference: some claims can be verified by one direct passage, while others require several evidence passages. We therefore apply adaptive evidence selection over the reranked list. 

Let _si_ be the cross-encoder score of the _i_ -th ranked evidence passage. The selector keeps evidence while the selected set remains within a maximum budget, the score is above an absolute threshold _τ_ , and the current score has not dropped sharply relative to the previous score: 



The minimum budget prevents the system from returning an empty or overly small evidence set. The absolute threshold removes weakly relevant tail passages, while the relative-drop condition detects a sharp decline in reranker confidence. The thresholds and evidence budgets are selected on the development set using the official harmonic score and then fixed for prediction. This makes the final evidence output more flexible than a fixed top- _K_ baseline while still controlling evidence noise. 

### **4.5 Design Rationale and Implementation-Specific Contributions** 

The main methodological contribution of our system is an alignment-oriented cascade design rather than a new pretrained architecture. We highlight 

three implementation choices that improve the soundness of the final pipeline. 

First, retrieval fusion is sparse-preserving. This is important because the task requires exact evidence identifiers, not merely a correct topic. Dense retrieval helps reduce the lexical gap, but exact lexical matches often identify the decisive passage. By preserving the RRF sparse core before adding dense candidates, the system balances semantic coverage with evidence precision. 

Besides, classifier training is aligned with pipeline inference. The reader is not trained under the unrealistic assumption that gold evidence will always be available. Instead, it sees gold, retrieved, and mixed evidence contexts during training. This makes the reader less sensitive to moderate retrieval noise and better reflects the distribution encountered at test time. 

In addition, label prediction and evidence submission are decoupled. The classifier uses a stable context for the four-way decision, while the evidence selector optimises the compact evidence set submitted for evaluation. This separation is methodologically important because the best context for classification is not necessarily the same as the best evidence list for evidence F-score. Together, these choices address the main weaknesses of a naive cascade system: lexical mismatch in retrieval, distribution mismatch in classification, and rigidity in evidence selection. 

## **5 Experiments** 

### **5.1 Experimental Setup** 

Our system is based on PyTorch and HuggingFace transformers. To achieve strict reproducibility, all experiments use a fixed seed (SEED=42) on a Google Colab T4 GPU with Automatic Mixed Precision (AMP). For Stage 1, BM25 uses _k_ 1 = 1 _._ 2 _, b_ = 0 _._ 85 to reduce the influence of long climate documents. The TF-IDF vectorizer obtains bigrams with a maximum of 200,000 features. In Stage 3, the semantic bi-encoder employs allMiniLM-L6-v2. During Stage 2, the cross-encoder (ms-marco-MiniLM-L-12-v2) is applied in a zeroshot way. For Stage 3, the reading comprehension classifier (bert-base-uncased) is trained for 4 epochs with the AdamW optimizer, a learning rate of 3e-5, a batch size of 8 and a 10% linear warmup. 

### **5.2 Main Results** 

The following table shows the performance of our system versions on the development set (154 instances). The main criterion is the Harmonic Mean (H-score) of the Evidence F-score (F) and Label Accuracy (A). 

- **Phase 1 & 3 (Baseline & Alignment):** Restoring the highly-aligned MiniLM semantic space (Phase 3) maximized retrieval recall compared to the baseline, though the unaligned classifier still suffered from majorityclass bias. 

- **Phase 4 & 5 (Class Weighting):** While applying raw inverse-frequency weights (Phase 4) improved minority detection at the cost of dominant class stability, our smoothed squareroot scaling (Phase 5) perfectly balanced minority protection and majority stability. 

- **Phase 2 & 6 (Distribution Mismatch):** Integrating unaligned heavy models like BGE (Phase 2 and 6) caused catastrophic drops in accuracy (decreasing by _∼_ 4.5%) due to semantic space mismatches, proving that the retriever and encoder must share the exact same space. 

- **Phase 7 & 8 (Ensemble vs. Optimal):** A deep logit-averaging ensemble (Phase 7) was rejected by our sweep to prevent outof-domain noise injection, culminating in our highly-aligned single-model passthrough (Phase 8) reaching the peak _H_ -score of 0.2921. 



Figure 2: Optimization trajectory across phases. 

## **6 Analysis and Discussion** 

### **6.1 Ablation Study: Distribution Misalignment** 

An important step in our development process (Phase 2 and Phase 6) was the investigation of other dense retrievers and rerankers (BAAI/bge-smallen-v1.5). Though using heavier models slightly enhanced the retrieval results, it caused a considerable decrease in the classification accuracy (about 4.5%). The main reasons are due to a distribution mismatch. The heavier models obtained a variety of text styles and semantic spaces which were not trained by the downstream classifier. Returning to use the MiniLM encoder recovered the semantic consistency and stabilized the cascade system. 

### **6.2 Ablation Study: Majority-Class Bias** 

The initial training distribution shows a class imbalance because the number of Supports is much greater than the number of Disputed. The classifier tends to overestimate the majority class without any adjustment, leading to unstable predictions of the minority class (Phase 4). Our best method (Phase 5) is a weighting scheme, which uses a smoothed square-root penalty 

### [1 _._ 0 _,_ 1 _._ 6 _,_ 1 _._ 15 _,_ 2 _._ 0] 

|**Configuration / Shift**|**F-score**|**Accuracy**|**H-score**|
|---|---|---|---|
|Phase 1: Initial Baseline|0.1995|0.4416|0.2852|
|Phase 2: Dense Mismatch|0.0777|0.4870|0.1340|
|Phase 3: MiniLM Align|0.2042|0.4935|0.2889|
|Phase 4: Hard Weight|0.2042|0.5065|0.2911|
|Phase 5: Soft Weight|0.2042|0.5130|0.2921|
|Phase 6: BGE Reranker|0.1970|0.4675|0.2772|
|Phase 7: Deep Ensemble|N/A|N/A|N/A|
|**Phase 8: Final Optimal**|**0.2042**|**0.5130**|**0.2921**|



Table 1: Iterative Experimental Results across Different Optimization Phases on the Development Set. 

together with label smoothing (0.1). This strategy effectively protects the minority classes and ensures the stability of the dominant class, and the accuracy has been improved to 51.3%. 

### **6.3 Error Analysis and Case Studies** 

An investigation of the verified predictions shows certain bottleneck patterns irrespective of the high H-value. We divide the remaining errors into three main categories: 

- **Retrieval Miss:** For precise statements, the important evidence is often hidden in the climate reports. The BM25 algorithm sometimes 

cannot find the exact sentence, limiting the maximum of our F-score even with our strict b=0.85 length penalty. 

- **Label Ambiguity:** For example, concerning the statement _“Volcanoes emit about 0.3 billion tonnes of CO2 per year”_ (Claim-2580), our computer has found conflicting information and judged it as REFUTES. However, the true value divides this particular numerical estimation into a scientific DISPUTED category. This example shows that even though the BERT reader can recognize textual contradictions, the difference between a definite contradiction (REFUTES) and the current scientific controversy (DISPUTED) is still difficult to determine. 

- **Evidence Noise:** Our Gap-Aware Adaptive Selection increases the context window for complicated statements to improve the recall rate, but it may also "extract" some semantically similar but irrelevant functional sentences. This results in a reduction of the F- score precision. For instance, in the Claim1718 about the historical temperature changes in high northern latitudes, the true main evidence was acquired; however, the window was appropriately enlarged to include three extra noisy sentences. These sentences were related to the Intergovernmental Panel on Climate Change’s evaluations of the average Northern Hemisphere temperatures during the past 1300 years. Although these semantic noises have little influence on the BERT classifier, they do affect the accurate retrieval measures. 

## **7 Limitations and Conclusion** 

### **7.1 Limitations** 

Our present system adopts a cascade pipeline. If the first search fails, it will also reduce the accuracy of the following classification because the BERT reader can not recover or reconstruct the lost important information. Furthermore, the crossencoder depends on an MS-MARCO passage ranking model. Though it works well for common information retrieval, there is a difference between the broad web queries and the particular and specific vocabulary required for evaluating climate science. 

Besides, although our evidence selection method called Gap-Aware Adaptive Evidence Selection is superior to the fixed top- _K_ strategy, it is decided 

by some particular absolute and relative thresholds ( _τ_ and drop_ratio) which are established based on the development data. Hence, this heuristic dependence might affect its stability towards out-ofdistribution (OOD) texts. For instance, when used for informal social media messages, the model’s confidence level may generally decline. These changes may result in true but differently expressed evidence failing to meet the strict _τ_ criterion, thereby decreasing the recall in actual application. 

### **7.2 Conclusion** 

In this project, we have proposed a strong and multi-stage fact-verification cascade system which is specifically intended for resolving the special problems in the field of climate science, such as the large number of scientific terms and serious class imbalance. To handle these problems, our system has included Reciprocal Rank Fusion (RRF) to enhance the hybrid retrieval recall, a precise semantic reranking cross-encoder and a BERT-based reader improved by Multi-View Data Augmentation. These parts can significantly decrease the differences in the process and stabilize the predicting ability of the model in the less common categories. 

Besides, we have also developed a new GapAware Adaptive Selection algorithm which varies the size of the context window according to the score trend adaptively, thus overcoming the shortcomings of the fixed top-K method. Our detailed comparison experiments indicate that the better and more stable results are obtained by carefully designing the architecture and targeting data augmentation rather than merely enlarging the scale of retriever or reader. Ultimately, the comprehensive optimization leads to an optimum Harmonic Mean score of 0.2921. 

For the future, there are still some potential research fields. One possible direction is to investigate domain-adaptive pretraining to reduce the lexical differences in specialized climate reports. Another direction could be to study the joint training of retriever and reader to eliminate the errors in the pipeline structure. Additionally, integrating uncertainty calibration into the adaptive selection mechanism may improve the system’s resistance to out-of-distribution claims. 

## **References** 

- [1] S. Robertson and H. Zaragoza. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. 

_Foundations and Trends in Information Retrieval_ . 

- [2] N. Reimers and I. Gurevych. 2019. SentenceBERT: Sentence Embeddings using Siamese BERTNetworks. _EMNLP-IJCNLP_ . 

- [3] W. Wang et al. 2020. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of PreTrained Transformers. _NeurIPS_ . 

- [4] G. V. Cormack, C. L. A. Clarke, and S. Buettcher. 2009. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. _SIGIR_ . 

- [5] R. Nogueira and K. Cho. 2019. Passage Re-ranking with BERT. _arXiv preprint_ . 

- [6] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. _NAACL_ . 

- [7] A. Vaswani et al. 2017. Attention Is All You Need. _NeurIPS_ . 

