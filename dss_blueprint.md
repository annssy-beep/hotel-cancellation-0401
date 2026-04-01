# Revenue-aware Hotel DSS Blueprint

## 1. Current Baseline From Your Draft + Notebook

### Research positioning
- Core claim: this study is not a better classifier paper, but a revenue-aware decision support artifact paper.
- Methodology: DSR.
- Kernel theory: TTF for design requirements, ROMC for dashboard/interface organization.
- Artifact goal: `prediction -> profit-aware decision -> explanation -> stakeholder action guidance`.

### Modeling baseline to report
- Final operational logic in the notebook is currently:
  - `global XGB + hotel-specific window + hotel-specific threshold`
  - `City`: `window=12`, median threshold `0.15`
  - `Resort`: `window=9`, median threshold `0.12`
- This matches the notebook conclusion better than “separate local model for each hotel type”.
- If you keep the paper text saying “city/resort context-aware modeling”, phrase it as:
  - different temporal windows by hotel type
  - hotel-type-specific decision thresholds
  - hotel-type-specific explanation and action views
  - while training still benefits from a global learning base

### Fold-level reference numbers already available
- `City Hotel`
  - mean threshold: `0.1492`
  - median threshold: `0.15`
  - last test period `2017-08`: `ROC-AUC 0.8099`, `F1 0.6633`, `mod_EMPC 38.7503`
- `Resort Hotel`
  - mean threshold: `0.1356`
  - median threshold: `0.12`
  - last test period `2017-08`: `ROC-AUC 0.8663`, `F1 0.7208`, `mod_EMPC 88.6960`

### SHAP top features currently visible in notebook
- Common high-impact variables across both hotels:
  - `lead_time`
  - `market_segment_Online TA`
  - `required_car_parking_spaces`
  - `total_of_special_requests`
  - `country_group_PRT`
  - `country_group_EU`
  - `adr`
  - `booking_changes`
  - `customer_type_Transient`
  - `expected_revenue`
- This is useful for the transparency section because it supports “stable explanation structure”.

## 2. Recommended Paper Logic

### Clean theoretical storyline
Use the following logic consistently:

1. Hotel cancellation is a revenue-management problem, not just a prediction problem.
2. AUC-only evaluation is insufficient because intervention has cost and recovered value is heterogeneous.
3. H-EMPC converts prediction into a profit-aware intervention rule.
4. Hotel context matters because city and resort differ in booking rhythm and revenue logic.
5. Stakeholders need explanation and actionability, so the artifact must be a DSS, not just a model.

### How to present the artifact contribution
State the artifact as four connected modules:
- Predictive engine: XGB probability scoring
- Decision engine: H-EMPC thresholding and prioritization
- Explanation engine: SHAP global + local explanation
- Narrative engine: LLM + single-book RAG for RM-grounded interpretation

### Suggested clarification for the draft
Your current draft says “city vs resort each have best window” but the notebook also says “global model + hotel-specific decision rule may be more practical”.

To avoid contradiction, write it like this:
- The DSS adopts a context-aware decision configuration rather than fully isolated model silos.
- Specifically, a shared global learning base is retained for robustness, while hotel-type-specific rolling windows, thresholds, and explanation views are used at deployment.

That keeps the notebook result and DSR artifact logic aligned.

## 3. What To Report in the Explainability Section

### 3.1 Global SHAP
For each hotel type, report:
- top 10 mean absolute SHAP features
- brief interpretation of direction from beeswarm/dependence plots
- overlap of top features across hotel types
- differences in feature salience between city and resort

### 3.2 Monthly overall SHAP summary
This is the key bridge from model output to dashboard.

For each month and hotel type, generate:
- monthly predicted cancellation rate
- previous-month predicted cancellation rate
- month-over-month delta
- top positive SHAP drivers of cancellation
- top protective SHAP drivers
- top high-revenue risky segments
- top at-risk reservations above threshold

Suggested monthly aggregation fields:
- `month`
- `hotel_type`
- `n_bookings`
- `pred_cancel_rate`
- `prev_pred_cancel_rate`
- `delta_pred_cancel_rate`
- `threshold`
- `n_intervention_targets`
- `sum_expected_revenue_targeted`
- `avg_opportunity_score`
- `top_risk_factors`
- `top_stable_factors`
- `top_segments`

### 3.3 Segment-level drill-down
Allow filters for:
- market segment
- distribution channel
- customer type
- country group
- lead-time bucket
- ADR band
- LOS band
- repeated guest vs first-time guest

Then recompute or pre-aggregate:
- segment predicted cancel rate
- segment SHAP top factors
- segment total exposed revenue
- recommended interventions by stakeholder

### 3.4 Local explanation
For reservation-level review:
- show predicted probability
- threshold
- risk excess above threshold
- expected revenue
- opportunity score
- top 5 positive SHAP contributors
- top 3 protective contributors
- stakeholder-specific action hints

Use SHAP as the main explanation method.
LIME can stay as an auxiliary robustness appendix if needed, but for the actual dashboard SHAP is cleaner and more defensible.

## 4. Decision Logic To Put Into the DSS

### Core scoring fields
Per reservation:
- `cancel_prob`
- `hotel_type`
- `threshold`
- `predicted_intervention` = `cancel_prob >= threshold`
- `expected_revenue`
- `risk_excess` = `max(cancel_prob - threshold, 0)`
- `opportunity_score` = `risk_excess * expected_revenue`

### Priority tiers
Use a practical decision tier instead of binary only:
- `Tier 1 Critical`: high probability and high revenue
- `Tier 2 Revenue-sensitive`: near-threshold but high revenue
- `Tier 3 Operational risk`: high probability but lower revenue
- `Tier 4 Monitor`: below threshold but trend worsening

This is better for stakeholders than only showing cancel / not cancel.

### Month-over-month managerial indicators
At dashboard summary level:
- predicted cancellation rate change
- targeted intervention volume
- revenue exposure of targeted bookings
- top worsening segments
- top improving segments
- expected prevented revenue under intervention policy

## 5. Stakeholder View Design

### RM Manager
Needs:
- thresholded intervention queue
- revenue exposure
- segment prioritization
- overbooking / deposit / reconfirmation guidance

Show:
- monthly risk summary
- top opportunity reservations
- top revenue-sensitive segments
- threshold vs expected gain panel

### Reservations / Rooms Manager
Needs:
- operationally actionable list
- contact / reconfirmation / payment follow-up priorities

Show:
- daily or monthly intervention list
- top reservation-level drivers
- guests requiring manual review
- bookings near threshold with high revenue

### Marketing / Sales
Needs:
- which segments are unstable
- which channel/customer groups need incentive or message redesign

Show:
- segment filter view
- channel-wise risk heatmap
- campaign suggestion panel
- OTA/direct/group/customer-type summaries

### GM / Finance
Needs:
- executive summary
- expected revenue at risk
- expected impact of intervention policy
- comparison against prior month

Show:
- KPI cards
- trend chart
- exposure by hotel type and segment
- concise LLM narrative with citations

## 6. ROMC-Aligned Dashboard Structure

### Representation
- KPI cards: predicted cancel rate, MoM change, targeted bookings, revenue exposure
- trend lines: monthly predicted cancel rate, revenue exposure, intervention volume
- SHAP summary bar / beeswarm
- segment heatmap
- reservation table with rank, opportunity score, major drivers
- stakeholder narrative panel

### Operations
- hotel type switch: city / resort
- month selector
- stakeholder mode selector
- segment filters
- booking-level search by reservation ID or user ID
- sort by probability / expected revenue / opportunity score
- drill-down from month -> segment -> reservation

### Memory Aids
- previous-month comparison cards
- saved report snapshot
- bookmarked segments / reservations
- export summary as PDF/CSV
- “why changed since last month” text panel

### Control Mechanisms
- role-based landing page
- guided query prompts
- toggle for probability / threshold / opportunity score view
- LLM explanation button with cited RM snippets
- admin setting for threshold policy and alert sensitivity

## 7. Recommended Web Dashboard Flow

### Main navigation
1. Overview
2. City Hotel
3. Resort Hotel
4. Segment Explorer
5. Reservation Explorer
6. RM Copilot
7. Admin / Model Info

### Screen flow
1. User logs in and selects role.
2. System loads role-optimized dashboard.
3. User chooses hotel type and month.
4. System shows monthly KPI cards and SHAP-based key drivers.
5. User filters by segment or channel.
6. System updates risk profile, exposed revenue, and recommended actions.
7. User drills down to individual reservations.
8. LLM explains the result in RM language and suggests role-specific responses.

## 8. LLM + RAG Design

### Intended role of the LLM
Do not let the LLM predict.
The LLM should only:
- interpret model outputs
- translate SHAP findings into RM language
- tailor recommendations to stakeholder roles
- explain trade-offs and likely actions

### Single-book RAG is enough if scope is narrow
Use the PDF only for:
- RM terminology grounding
- intervention rationale
- segment/channel/seasonality logic
- pricing, distribution, overbooking, deposit, demand management concepts

### Retrieval units
Chunk the PDF by:
- chapter
- section
- subsection

Recommended chunk size:
- around `600-1000` tokens
- overlap `100-150` tokens

Metadata per chunk:
- `chapter`
- `section_title`
- `page_start`
- `page_end`
- `topic_tags` such as `overbooking`, `distribution`, `pricing`, `segment`, `forecasting`

### Prompt contract
System prompt should say:
- You are an RM decision-support copilot.
- You must not invent model results.
- Use only provided model outputs and retrieved excerpts.
- Separate `Observed model signal`, `RM interpretation`, and `Suggested action`.
- Cite the retrieved chunk IDs or chapter/page references.

### Recommended output format
- `Observed signal`
- `Main drivers`
- `RM interpretation`
- `Recommended actions by stakeholder`
- `Cautions / assumptions`
- `Source grounding`

### Cheap API option
Your idea is reasonable:
- use `gpt-5-mini` for narrative generation
- keep context small by passing:
  - monthly metrics
  - top SHAP features
  - segment filter results
  - 2 to 5 retrieved book chunks

### Minimal API settings you will need
- `OPENAI_API_KEY`
- model name
- embedding model for RAG
- vector store path or DB config

Practical starter stack:
- app: `Streamlit` or `Next.js`
- backend: `FastAPI`
- vector DB: `FAISS` or `Chroma`
- LLM API: OpenAI Responses API
- embeddings: one small embedding model for the PDF chunks

## 9. Suggested Data Products To Materialize

Create these tables/files so the dashboard stays fast:

### Reservation scoring table
- one row per booking
- hotel type, month, segment fields
- prediction outputs
- threshold outputs
- SHAP top features
- opportunity score

### Monthly summary table
- one row per month per hotel type
- KPI aggregates
- MoM delta
- top SHAP factors
- top risky segments

### Segment summary table
- one row per month x hotel type x segment key
- predicted cancel rate
- exposure
- SHAP summary
- intervention counts

### RAG index
- chunked book passages with metadata and embeddings

## 10. Implementation Order

### Phase 1. Analysis outputs
- Freeze the final modeling logic:
  - `City = global XGB, window 12, threshold 0.15`
  - `Resort = global XGB, window 9, threshold 0.12`
- Export scored reservation-level dataset.
- Compute SHAP values for deployment sample or monthly scoring set.
- Build monthly and segment aggregate tables.

### Phase 2. Dashboard MVP
- Overview page
- hotel-type page
- segment explorer
- reservation drill-down
- downloadable monthly report

### Phase 3. LLM copilot
- PDF chunking and embeddings
- retrieval service
- structured prompt templates by stakeholder
- answer panel with citations

### Phase 4. Evaluation
- performance: ROC-AUC, Brier, EMPC, H-EMPC
- transparency: SHAP stability across folds/months
- fit: TTF interview/survey
- usability: stakeholder walkthrough

## 11. What You Can Write Directly Into the Paper Now

### Implementation section
You can already say:
- The DSS uses a global XGBoost predictive base trained on rolling windows.
- Window size is contextually configured by hotel type: 12 months for city, 9 months for resort.
- Intervention thresholds are optimized under H-EMPC and deployed as hotel-type-specific decision rules.
- SHAP provides global and local explanations.
- A lightweight LLM module grounded in one hospitality RM textbook generates stakeholder-facing narratives and action suggestions.

### Evaluation section
You can already fill:
- City and Resort final thresholds
- last-period reference performance
- SHAP top features
- logic for transparency and usefulness evaluation

## 12. Concrete Build Recommendation

If you want the fastest path to a working artifact for paper/demo, build this stack:
- frontend: `Streamlit`
- backend logic: same Python environment as notebook
- explanation: `shap`
- monthly tables: `pandas`
- RAG: `langchain` or direct OpenAI SDK + `FAISS`
- narrative model: `gpt-5-mini`

This is enough for a publishable prototype without overengineering.

## 13. API / Config You Will Need Later

Set at minimum:

```bash
export OPENAI_API_KEY=your_key_here
```

Optional app config:

```bash
export OPENAI_MODEL=gpt-5-mini
export EMBEDDING_MODEL=text-embedding-3-small
export VECTOR_DB_PATH=./rag_index
```

## 14. Immediate Next Work I Recommend

1. Refactor the notebook into reusable scoring functions.
2. Export `city` and `resort` reservation scoring tables with SHAP values.
3. Build monthly summary and segment summary tables.
4. Stand up a simple dashboard MVP in Streamlit.
5. Add the RAG-backed stakeholder narrative panel last.

## 15. Important Correction To Keep In Mind

Your message said “best is XGB City window=9, Resort window=12”.
But the current notebook and draft tables indicate the opposite:
- `City -> window 12`
- `Resort -> window 9`

For the paper and system design, keep those exact values unless you rerun the notebook and obtain new final results.
