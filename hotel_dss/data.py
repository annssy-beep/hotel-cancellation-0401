from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = BASE_DIR / "hotel_bookings_preprocessed_update(0129).csv"
MONTH_COLUMN = "month_period"
ROLE_OPTIONS = [
    "Revenue Manager",
    "Reservations / Rooms Manager",
    "Marketing / Sales",
    "GM / Finance",
]
FILTER_COLUMNS = [
    "market_segment_label",
    "customer_type_label",
    "country_group_label",
]

LABEL_PREFIXES = {
    "market_segment_": "market_segment_label",
    "distribution_channel_": "distribution_channel_label",
    "customer_type_": "customer_type_label",
    "country_group_": "country_group_label",
    "deposit_type_": "deposit_type_label",
}

NON_FEATURE_COLUMNS = {
    "arrival_date",
    "booking_date",
    MONTH_COLUMN,
    "booking_id",
    "hotel_type",
    "lead_time_bucket",
    "adr_band",
    "los_band",
    "market_segment_label",
    "distribution_channel_label",
    "customer_type_label",
    "country_group_label",
    "deposit_type_label",
    "pred_prob",
    "threshold",
    "predicted_intervention",
    "risk_excess",
    "opportunity_score",
    "risk_tier",
    "decision_action",
}


def decode_one_hot_label(df: pd.DataFrame, prefix: str, default_label: str = "Unknown") -> pd.Series:
    columns = [col for col in df.columns if col.startswith(prefix)]
    if not columns:
        return pd.Series(default_label, index=df.index)
    values = df[columns].idxmax(axis=1).str.replace(prefix, "", regex=False)
    active = df[columns].max(axis=1) > 0
    return values.where(active, default_label)


def prepare_dataframe(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    df[MONTH_COLUMN] = df["arrival_date"].dt.to_period("M").astype(str)
    df["booking_id"] = [f"BK-{idx:06d}" for idx in range(len(df))]
    df["hotel_type"] = np.where(df["hotel_City Hotel"] == 1, "City Hotel", "Resort Hotel")

    for prefix, label_col in LABEL_PREFIXES.items():
        df[label_col] = decode_one_hot_label(df, prefix)

    df["lead_time_bucket"] = pd.cut(
        df["lead_time"],
        bins=[-1, 7, 30, 90, 180, np.inf],
        labels=["0-7d", "8-30d", "31-90d", "91-180d", "180d+"],
    ).astype(str)

    df["adr_band"] = pd.qcut(df["adr"], q=4, labels=["Low", "Mid-Low", "Mid-High", "High"], duplicates="drop").astype(str)
    df["los_band"] = pd.cut(
        df["length_of_stay"],
        bins=[0, 1, 3, 7, np.inf],
        labels=["1 night", "2-3 nights", "4-7 nights", "8+ nights"],
        include_lowest=True,
    ).astype(str)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in NON_FEATURE_COLUMNS and col != "is_canceled"]


def filter_scored_dataframe(
    scored_df: pd.DataFrame,
    hotel_type: str,
    selected_month: str,
    filters: dict[str, list[str]],
) -> pd.DataFrame:
    df = scored_df.copy()
    if hotel_type != "All":
        df = df[df["hotel_type"] == hotel_type]
    if selected_month:
        df = df[df[MONTH_COLUMN] <= selected_month]
    for column, values in filters.items():
        if values:
            df = df[df[column].isin(values)]
    return df


def build_monthly_overview(scored_df: pd.DataFrame, selected_month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    month_df = scored_df[scored_df[MONTH_COLUMN] == selected_month].copy()
    unique_months = sorted(scored_df[MONTH_COLUMN].unique())
    if selected_month not in unique_months:
        return month_df, pd.DataFrame()
    idx = unique_months.index(selected_month)
    if idx == 0:
        return month_df, pd.DataFrame()
    previous_df = scored_df[scored_df[MONTH_COLUMN] == unique_months[idx - 1]].copy()
    return month_df, previous_df


def build_segment_summary(scored_df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if scored_df.empty:
        return pd.DataFrame()
    summary = (
        scored_df.groupby(dimension, as_index=False)
        .agg(
            bookings=("booking_id", "count"),
            actual_cancel_rate=("is_canceled", "mean"),
            pred_cancel_rate=("pred_prob", "mean"),
            targeted_bookings=("predicted_intervention", "sum"),
            revenue_exposure=("expected_revenue", "sum"),
            opportunity_score=("opportunity_score", "sum"),
        )
        .sort_values(["pred_cancel_rate", "opportunity_score"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return summary
