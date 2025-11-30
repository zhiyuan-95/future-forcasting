# future-forcasting

### **Forecasting Societal Events Using Neural Temporal Models Enhanced by LLM-Derived Event Semantics**

Societies continuously produce political, social, and economic events that influence stability, risk, and development. Timely forecasting of such events—protests, conflicts, policy shifts, or economic disruptions—is crucial for governments, NGOs, and institutions. Traditional forecasting systems rely on limited structured datasets (e.g., ACLED, ICEWS, GDELT) and simplify events into predefined categories, losing important context and nuance. This project introduces a new framework that combines **neural temporal point processes** with **large language model (LLM)–derived semantic representations** to predict future societal events with significantly greater resolution and accuracy.

The proposed model treats each country as a **sequence of historical events**, where each event includes its timestamp, type, and an automatically generated impact score on a ([-10, 10]) scale. Using an LLM, the system extracts rich event summaries and semantic embeddings directly from news articles, enabling the model to capture factors such as actor motivations, underlying social tensions, tone, and consequences—information typically absent from standard datasets. These semantically enriched events are then fed into a **Neural Hawkes–style temporal model** (or Recurrent Marked Temporal Point Process), which is specifically designed to learn dependencies across time in event sequences.

The model jointly predicts three key properties of the next event:

1. **When** it will occur (time-to-event forecasting)
2. **What type** of event it will be (categorical classification)
3. **How impactful** it will be on the society (regression on the ([-10, 10]) scale)

This multi-task formulation enables a comprehensive understanding of future risks rather than simple binary or category-level forecasts. Moreover, the design naturally accommodates **country-specific embeddings** and can be extended to include cross-country influence through spatio-temporal graph models.

Evaluation will include accuracy on event type prediction, error in timing forecasts, impact prediction performance, and ablation studies that assess the contribution of LLM-derived semantics versus traditional handcrafted features. The model will be benchmarked against classical Hawkes processes and purely text-based LLM forecasting baselines.

This project is both **technically feasible** and **novel**. All required datasets are public, the LLM component is lightweight (summarization + scoring), and the temporal model is compact enough for efficient training. The result will be a unified forecasting system that improves upon existing political event prediction frameworks by incorporating both **temporal patterns** and **contextual understanding** of events. The project has strong potential for academic publication, practical application in risk analysis, and long-term extensibility into global forecasting, early warning systems, and international relations modeling.

This executive summary reflects a scalable, impactful, and research-worthy direction appropriate for a master’s capstone or early-stage academic research project.

## Detailed Research Proposal

See the full proposal in [`docs/Research_Proposal.md`](docs/Research_Proposal.md) for a comprehensive description of motivation, methodology, evaluation, and deliverables.
