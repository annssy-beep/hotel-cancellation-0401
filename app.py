from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from hotel_dss.data import (
    DEFAULT_CSV_PATH,
    FILTER_COLUMNS,
    MONTH_COLUMN,
    ROLE_OPTIONS,
    build_monthly_overview,
    build_segment_summary,
    filter_scored_dataframe,
    get_feature_columns,
    prepare_dataframe,
)
from hotel_dss.explain import get_global_shap_summary, get_local_shap_explanation
from hotel_dss.llm import build_brief_with_fallback, generate_role_narrative, llm_status
from hotel_dss.modeling import build_scored_dataset


st.set_page_config(
    page_title="Revenue-aware Hotel DSS",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

PDF_PATH = Path(__file__).resolve().parent / "revenue-management-for-the-hospitality-industry-2nbsped-1119790778-9781119790778_compress.pdf"


@st.cache_resource(show_spinner="Preparing models and scored booking data...")
def load_runtime(csv_path: str):
    df = prepare_dataframe(csv_path)
    feature_cols = get_feature_columns(df)
    runtime = build_scored_dataset(df, feature_cols)
    return runtime


def render_metrics(month_df: pd.DataFrame, previous_df: pd.DataFrame):
    prev_rate = previous_df["pred_prob"].mean() if len(previous_df) else 0.0
    curr_rate = month_df["pred_prob"].mean() if len(month_df) else 0.0
    prev_targets = int(previous_df["predicted_intervention"].sum()) if len(previous_df) else 0
    curr_targets = int(month_df["predicted_intervention"].sum()) if len(month_df) else 0
    prev_exposure = previous_df["expected_revenue"].sum() if len(previous_df) else 0.0
    curr_exposure = month_df["expected_revenue"].sum() if len(month_df) else 0.0
    prev_opp = previous_df["opportunity_score"].sum() if len(previous_df) else 0.0
    curr_opp = month_df["opportunity_score"].sum() if len(month_df) else 0.0

    cols = st.columns(4)
    cols[0].metric(
        "Predicted cancel rate",
        f"{curr_rate:.1%}",
        f"{(curr_rate - prev_rate):+.1%}",
    )
    cols[1].metric(
        "Intervention targets",
        f"{curr_targets:,}",
        f"{curr_targets - prev_targets:+,}",
    )
    cols[2].metric(
        "Revenue exposure",
        f"{curr_exposure:,.0f}",
        f"{curr_exposure - prev_exposure:+,.0f}",
    )
    cols[3].metric(
        "Intervention opportunity",
        f"{curr_opp:,.0f}",
        f"{curr_opp - prev_opp:+,.0f}",
    )
    st.caption(
        "Intervention opportunity = monthly sum of positive risk excess above threshold multiplied by expected booking revenue. "
        "It is a prioritization signal, not realized revenue."
    )


def build_summary_payload(role: str, hotel_type: str, selected_month: str, month_df: pd.DataFrame, scored_df: pd.DataFrame):
    if hotel_type == "All" and len(month_df):
        effective_hotel_type = month_df["hotel_type"].mode().iloc[0]
    else:
        effective_hotel_type = hotel_type

    top_drivers = []
    if effective_hotel_type != "All" and len(month_df):
        shap_summary = get_global_shap_summary(
            model_bundle=scored_df.attrs["model_bundle_map"][effective_hotel_type],
            subset_df=month_df[month_df["hotel_type"] == effective_hotel_type],
            feature_cols=scored_df.attrs["feature_cols"],
            top_n=4,
        )
        top_drivers = shap_summary["feature_label"].tolist() if not shap_summary.empty else []

    return {
        "role": role,
        "hotel_type": effective_hotel_type,
        "month": selected_month,
        "pred_cancel_rate": round(float(month_df["pred_prob"].mean()), 4) if len(month_df) else 0.0,
        "revenue_exposure": round(float(month_df["expected_revenue"].sum()), 2) if len(month_df) else 0.0,
        "opportunity_score": round(float(month_df["opportunity_score"].sum()), 2) if len(month_df) else 0.0,
        "targeted_bookings": int(month_df["predicted_intervention"].sum()) if len(month_df) else 0,
        "top_segments": (
            month_df.groupby("market_segment_label")["pred_prob"]
            .mean()
            .sort_values(ascending=False)
            .head(2)
            .index.tolist()
            if len(month_df)
            else []
        ),
        "top_drivers": top_drivers,
        "top_risky_bookings": (
            month_df.sort_values("opportunity_score", ascending=False)["booking_id"].head(3).tolist()
            if len(month_df)
            else []
        ),
    }


def render_ai_brief(role: str, hotel_type: str, selected_month: str, month_df: pd.DataFrame, scored_df: pd.DataFrame):
    st.subheader("AI Decision Brief")
    status = llm_status()
    summary_payload = build_summary_payload(role, hotel_type, selected_month, month_df, scored_df)

    top = st.columns([1.1, 1.1, 0.9])
    with top[0]:
        st.markdown("**Current signal**")
        st.write(generate_role_narrative(summary_payload))
    with top[1]:
        st.markdown("**Main drivers**")
        drivers = summary_payload["top_drivers"][:3]
        st.write(" | ".join(drivers) if drivers else "No SHAP evidence for the current filter.")
    with top[2]:
        generate = st.button("Refresh AI Strategy", use_container_width=True, key=f"brief_btn_{role}_{hotel_type}_{selected_month}")
        if not status["ready"]:
            st.caption(status["message"])

    result = None
    state_key = f"brief_result_{role}_{hotel_type}_{selected_month}"
    if generate:
        with st.spinner("Generating intervention strategy..."):
            result = build_brief_with_fallback(
                summary_payload=summary_payload,
                pdf_path=str(PDF_PATH),
                use_embeddings=True,
                top_k=2,
            )
        st.session_state[state_key] = result
    else:
        result = st.session_state.get(state_key)

    if result is None:
        st.info("Click `Refresh AI Strategy` to generate a compact intervention brief for the current month, hotel, and filters.")
        return

    st.markdown(result["llm_text"] or result["narrative_preview"])
    if result.get("error"):
        st.caption(f"Fallback active: {result['error']}")
    citations = []
    for chunk in result.get("retrieved_chunks", [])[:2]:
        page = chunk.get("page_start", "?")
        chapter = chunk.get("chapter", "Unknown")
        citations.append(f"{chapter}, p.{page}")
    if citations:
        st.caption("Source: " + "; ".join(citations))

    with st.expander("View grounding details"):
        st.caption(f"mode: {result['mode']}")
        for idx, chunk in enumerate(result.get("retrieved_chunks", [])[:2], start=1):
            st.markdown(f"**Source {idx} | {chunk.get('chapter', 'Unknown')} | page {chunk.get('page_start', '?')}**")
            st.write(chunk.get("text", "")[:500] + ("..." if len(chunk.get("text", "")) > 500 else ""))


def render_overview(scored_df: pd.DataFrame, hotel_type: str, selected_month: str, role: str):
    st.subheader("Monthly Overview")
    month_df, previous_df = build_monthly_overview(scored_df, selected_month)
    render_metrics(month_df, previous_df)
    render_ai_brief(role, hotel_type, selected_month, month_df, scored_df)

    trend = (
        scored_df.groupby([MONTH_COLUMN, "hotel_type"], as_index=False)
        .agg(
            pred_cancel_rate=("pred_prob", "mean"),
            opportunity_score=("opportunity_score", "sum"),
            targeted_bookings=("predicted_intervention", "sum"),
        )
        .sort_values(MONTH_COLUMN)
    )
    trend_fig = px.line(
        trend,
        x=MONTH_COLUMN,
        y="pred_cancel_rate",
        color="hotel_type",
        markers=True,
        title="Monthly predicted cancellation rate",
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    exposure_fig = px.bar(
        trend,
        x=MONTH_COLUMN,
        y="opportunity_score",
        color="hotel_type",
        barmode="group",
        title="Monthly intervention opportunity",
    )
    st.plotly_chart(exposure_fig, use_container_width=True)

    if hotel_type == "All":
        st.info("SHAP explanation is shown when a specific hotel type is selected.")
        return

    shap_summary = get_global_shap_summary(
        model_bundle=scored_df.attrs["model_bundle_map"][hotel_type],
        subset_df=month_df,
        feature_cols=scored_df.attrs["feature_cols"],
        top_n=12,
    )
    if shap_summary.empty:
        st.warning("There is not enough data to compute SHAP for the selected filter.")
        return

    shap_fig = px.bar(
        shap_summary.sort_values("mean_abs_shap"),
        x="mean_abs_shap",
        y="feature_label",
        orientation="h",
        color="direction",
        title="Top SHAP drivers for the selected month",
    )
    st.plotly_chart(shap_fig, use_container_width=True)

    pos = shap_summary[shap_summary["direction"] == "Increase risk"]["feature_label"].head(5).tolist()
    neg = shap_summary[shap_summary["direction"] == "Protective"]["feature_label"].head(5).tolist()
    col1, col2 = st.columns(2)
    col1.markdown("**Main cancellation drivers**")
    col1.write(", ".join(pos) if pos else "None")
    col2.markdown("**Protective factors**")
    col2.write(", ".join(neg) if neg else "None")


def render_segment_explorer(scored_df: pd.DataFrame, hotel_type: str):
    st.subheader("Segment Explorer")
    dimension = st.selectbox("Segment dimension", FILTER_COLUMNS, index=0)
    segment_summary = build_segment_summary(scored_df, dimension)
    if segment_summary.empty:
        st.warning("No segment summary is available for the current filter.")
        return

    fig = px.bar(
        segment_summary.head(15),
        x=dimension,
        y="pred_cancel_rate",
        color="revenue_exposure",
        title=f"Predicted cancellation rate by {dimension}",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(segment_summary, use_container_width=True, hide_index=True)

    if hotel_type != "All":
        top_segment = segment_summary.iloc[0][dimension]
        st.caption(f"Highest-risk segment under the current filter: `{top_segment}`")


def render_reservation_explorer(scored_df: pd.DataFrame, hotel_type: str):
    st.subheader("Reservation Explorer")
    top_cases = scored_df.sort_values(
        ["opportunity_score", "pred_prob", "expected_revenue"],
        ascending=[False, False, False],
    ).head(100)
    st.dataframe(
        top_cases[
            [
                "booking_id",
                "hotel_type",
                MONTH_COLUMN,
                "pred_prob",
                "threshold",
                "risk_tier",
                "expected_revenue",
                "opportunity_score",
                "market_segment_label",
                "customer_type_label",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    booking_id = st.selectbox("Inspect booking", top_cases["booking_id"].tolist())
    row_df = top_cases[top_cases["booking_id"] == booking_id]
    if row_df.empty:
        return

    cols = st.columns(4)
    row = row_df.iloc[0]
    cols[0].metric("Cancel probability", f"{row['pred_prob']:.1%}")
    cols[1].metric("Threshold", f"{row['threshold']:.0%}")
    cols[2].metric("Expected revenue", f"{row['expected_revenue']:,.0f}")
    cols[3].metric("Opportunity score", f"{row['opportunity_score']:,.0f}")

    if hotel_type == "All":
        hotel_type = row["hotel_type"]

    local_shap = get_local_shap_explanation(
        model_bundle=scored_df.attrs["model_bundle_map"][hotel_type],
        row_df=row_df,
        feature_cols=scored_df.attrs["feature_cols"],
        top_n=10,
    )
    if local_shap.empty:
        st.warning("SHAP explanation could not be computed for the selected booking.")
        return

    shap_fig = px.bar(
        local_shap.sort_values("abs_shap"),
        x="shap_value",
        y="feature_label",
        orientation="h",
        color="direction",
        title=f"Local SHAP explanation: {booking_id}",
    )
    st.plotly_chart(shap_fig, use_container_width=True)
    st.dataframe(local_shap, use_container_width=True, hide_index=True)


def render_copilot(
    role: str,
    hotel_type: str,
    selected_month: str,
    month_df: pd.DataFrame,
    scored_df: pd.DataFrame,
):
    st.subheader("AI Brief Workspace")
    st.info("The AI intervention strategy is now surfaced in the Overview tab as the primary decision-support panel.")


def render_method(scored_df: pd.DataFrame):
    st.subheader("Artifact Logic")
    st.markdown(
        """
        - Predictive engine: global XGBoost with hotel-specific deployment windows
        - City Hotel: window `12`, threshold `0.15`
        - Resort Hotel: window `9`, threshold `0.12`
        - Decision engine: `cancel_prob`, `risk_excess`, `opportunity_score`
        - Explanation engine: SHAP global summary + reservation-level local explanation
        - Narrative engine: optional `gpt-5-mini + single-book RAG`
        """
    )
    st.markdown("**Current app backend**")
    backend_df = pd.DataFrame(
        [
            {
                "hotel_type": hotel_type,
                "model_backend": bundle["model_name"],
                "window": bundle["window"],
                "threshold": bundle["threshold"],
            }
            for hotel_type, bundle in scored_df.attrs["model_bundle_map"].items()
        ]
    )
    st.dataframe(backend_df, use_container_width=True, hide_index=True)
    st.markdown("**Current dataset scope**")
    st.write(
        scored_df.groupby("hotel_type")["booking_id"]
        .count()
        .rename("bookings")
        .reset_index()
    )


def main():
    st.title("Revenue-aware Decision Support System for Hotel RM")
    st.caption("DSR artifact MVP: prediction, profit-aware prioritization, SHAP evidence, and AI-supported intervention guidance")

    runtime = load_runtime(str(DEFAULT_CSV_PATH))
    scored_df = runtime["scored_df"]
    scored_df.attrs["model_bundle_map"] = runtime["model_bundle_map"]
    scored_df.attrs["feature_cols"] = runtime["feature_cols"]

    months = sorted(scored_df[MONTH_COLUMN].unique())
    default_month = months[-1]

    with st.sidebar:
        st.header("Controls")
        role = st.selectbox("Stakeholder role", ROLE_OPTIONS, index=0)
        hotel_type = st.selectbox("Hotel type", ["All", "City Hotel", "Resort Hotel"], index=0)
        selected_month = st.selectbox("Month", months[::-1], index=0)
        filters = {}
        for column in FILTER_COLUMNS:
            values = sorted(scored_df[column].dropna().unique().tolist())
            filters[column] = st.multiselect(column, values, default=[])
        st.caption(f"Default month: `{default_month}`")

    filtered_df = filter_scored_dataframe(
        scored_df,
        hotel_type=hotel_type,
        selected_month=selected_month,
        filters=filters,
    )
    month_df, previous_df = build_monthly_overview(filtered_df, selected_month)

    tabs = st.tabs(
        [
            "Overview",
            "Segment Explorer",
            "Reservation Explorer",
            "RM Copilot",
            "Method",
        ]
    )

    with tabs[0]:
        render_overview(filtered_df, hotel_type, selected_month, role)
    with tabs[1]:
        render_segment_explorer(filtered_df, hotel_type)
    with tabs[2]:
        render_reservation_explorer(filtered_df, hotel_type)
    with tabs[3]:
        render_copilot(role, hotel_type, selected_month, month_df, scored_df)
    with tabs[4]:
        render_method(scored_df)


if __name__ == "__main__":
    main()
