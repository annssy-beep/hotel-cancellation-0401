# Revenue-aware Hotel DSS MVP

## Run

```bash
streamlit run app.py
```

## What is implemented

- Global deployment logic with hotel-specific operating windows
- `City Hotel`: window `12`, threshold `0.15`
- `Resort Hotel`: window `9`, threshold `0.12`
- Reservation-level scoring:
  - `cancel_prob`
  - `threshold`
  - `predicted_intervention`
  - `risk_excess`
  - `opportunity_score`
  - `risk_tier`
- SHAP-based global and local explanations
- Stakeholder-oriented dashboard tabs
- LLM panel scaffold for `gpt-5-mini + single-book RAG`

## LLM and RAG

The app can run without LLM, but `RM Copilot` becomes fully active when you set the API key.

1. Install dependencies in [requirements.txt](/Users/annss/Desktop/DBI_LAB./HotelBooking/0401Prompt/requirements.txt).
2. Set:

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_MODEL=gpt-5-mini
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

3. Build the RAG assets from the textbook:

```bash
python3 scripts/build_rag_assets.py
```

This creates:
- `rag/rm_reference.txt`
- `rag/rm_chunks.jsonl`

4. In the dashboard:
- `Generate Brief: Keyword RAG` uses textbook chunk retrieval without embeddings.
- `Generate Brief: Embedding RAG` adds OpenAI embeddings and caches them locally in `rag/rm_embeddings.npz`.

## Notes

- The current app trains two cached models on startup and caches them in Streamlit.
- Default backend is `RandomForest` for sandbox stability.
- To try XGBoost in a less restricted local environment:

```bash
export HOTEL_DSS_MODEL=xgb
streamlit run app.py
```
- The dashboard uses the notebook's final operating logic rather than recomputing rolling thresholds every run.
- For paper writing, keep the deployment logic aligned with the notebook:
  - `City -> window 12`
  - `Resort -> window 9`
- OpenAI Responses API and embeddings are used for the LLM path. Official docs:
  - https://developers.openai.com/api/docs/models/compare
