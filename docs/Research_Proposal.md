# Forecasting Societal Events Using Neural Temporal Point Processes with LLM-Enhanced Event Representations

## 1. Motivation

Societies generate continuous streams of political, social, and economic events. Predicting future events—such as protests, policy shifts, outbreaks of violence, economic shocks, or other impactful occurrences—is important for early warning systems, policy planning, and risk assessment.

However, existing datasets used for forecasting (ACLED, ICEWS, GDELT) are often incomplete, heterogeneous in quality, and contain limited semantic information. Furthermore, events occur in sequences with strong temporal dependencies: one event often triggers another. We need a model that:

- captures **temporal dynamics** of past events,
- generalizes across **countries and social contexts**, and
- incorporates **semantic understanding** of events.

The goal is to output **multi-dimensional predictions**: when the next event will occur, what type it will be, and how strong its impact will be.

## 2. Research Question

**Can we build a unified model that predicts the timing, type, and impact of future socio-political events in a given country, using neural temporal point processes augmented with LLM-derived event embeddings?**

Sub-questions:

1. Do learned temporal dependencies meaningfully improve event forecasts over classical statistical baselines?
2. Does integrating LLM-derived semantic information enhance prediction accuracy?
3. How transferable are temporal patterns across different countries or regions?
4. Can an impact score in ([-10, 10]) capture the effect magnitude of events in a stable, model-learnable way?

## 3. Proposed Methodology

### 3.1 Data Collection

We use publicly available socio-political event datasets:

- **ACLED** – detailed political violence and protest events
- **ICEWS** – machine-coded political events
- **GDELT** – broad event extraction from global media

For each event, we retrieve associated news articles through common news APIs or GDELT’s raw text sources.

### 3.2 LLM-Based Event Enrichment

Each raw news article is processed by an LLM to extract:

1. **Event summary (semantic text)**
2. **Event attributes** such as actors, cause, sentiment, and context
3. **Impact score** in ([-10, 10])

The LLM’s final-layer embedding (or a projected version) becomes the **event representation vector (x_i)**, providing richer semantics than raw category labels alone.

### 3.3 Temporal Modeling: Neural Point Processes

We model each country as a marked temporal point process:

\[
 e_i = (t_i, k_i, s_i, x_i)
\]

Where:

- \(t_i\): time of event
- \(k_i\): event type (categorical)
- \(s_i\): impact label from the LLM
- \(x_i\): LLM semantic embedding

We encode sequences using a recurrent architecture:

\[
 h_i = \text{GRU}(v_i, h_{i-1})
\]

where

\[
 v_i = [\text{Emb}(k_i), \text{Emb}(country), f(\Delta t_i), s_i, x_i]
\]

From \(h_i\) we predict the next event components:

- **Event type**: \(p(k_{i+1} \mid h_i) = \text{softmax}(W_k h_i + b_k)\)
- **Time until next event**: modeled via RMTPP or Neural Hawkes intensity function, or simplified regression on \(\log \Delta t\)
- **Impact score**: \(\hat{s}_{i+1} = f_{\text{impact}}(h_i)\)

This yields a joint prediction: **when, what, and how severe** the next event will be.

### 3.4 Conditioning on Country or Region

We incorporate country embeddings and optionally dynamic covariates (GDP, population, regime type, elections). A graph-based extension (Graph Hawkes or spatio-temporal GNN) can model cross-country influence.

## 4. Evaluation

1. **Predictive accuracy**: event type prediction accuracy, timing error (e.g., RMSE on log time), and impact severity error (MAE/MSE)
2. **Calibration of event timing**: does predicted intensity match actual event frequencies?
3. **Generalization across countries**: train on some regions, test on unseen ones
4. **Ablations**: remove LLM-derived semantics, the GRU temporal encoder, country embeddings, or the impact-score head to isolate component contributions

## 5. Expected Contributions

1. A unified, multi-output event forecasting model predicting time, type, and impact
2. Integration of LLM-derived semantics into temporal point processes for richer event understanding
3. An event impact scoring system on a simple ([-10, 10]) scale derived from LLM labeling
4. A reusable pipeline for political forecasting, risk analysis, or social science modeling

## 6. Feasibility

- Data is publicly accessible (ACLED, ICEWS, GDELT).
- The LLM component requires only summarization and scoring.
- The temporal model is compact (GRU plus three heads) and efficient to train.
- The task fits a 6–9 month research or capstone schedule.

## 7. Timeline (suggested)

| Month | Task |
| --- | --- |
| 1 | Literature review, dataset selection, country choice |
| 2 | Data preprocessing, LLM labeling pipeline |
| 3–4 | Build baseline RMTPP / Hawkes model |
| 5 | Integrate LLM embeddings |
| 6 | Full model training and evaluation |
| 7 | Ablation studies |
| 8 | Write final report and prepare GitHub repo |
| 9 | (Optional) Submit to a workshop or conference |

## 8. Deliverables

- Runnable PyTorch codebase
- Curated event and impact dataset
- Multi-task event forecasting model
- Research-style report suitable for academic submission
- Optional visualization dashboard for future event probabilities

## 9. Conclusion

This proposal outlines a direction that blends **temporal reasoning** with **semantic event understanding** to forecast future societal events. The fusion of LLMs and neural temporal point processes creates a flexible framework for modeling complex human systems across countries and regions. It is suitable for a capstone project, master’s thesis, research collaboration, or a pre-publication research project.
