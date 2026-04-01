import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


WINDOW_BY_HOTEL = {
    "City Hotel": 12,
    "Resort Hotel": 9,
}

THRESHOLD_BY_HOTEL = {
    "City Hotel": 0.15,
    "Resort Hotel": 0.12,
}

DEFAULT_MODEL_BACKEND = os.getenv("HOTEL_DSS_MODEL", "rf").lower()

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "random_state": 42,
    "n_jobs": 1,
}


def get_latest_window_df(df: pd.DataFrame, train_months: int, date_col: str = "arrival_date") -> pd.DataFrame:
    tmp = df.copy()
    tmp["_month_period"] = tmp[date_col].dt.to_period("M")
    months = sorted(tmp["_month_period"].unique())
    latest_periods = months[-train_months:]
    out = tmp[tmp["_month_period"].isin(latest_periods)].copy()
    return out.drop(columns=["_month_period"], errors="ignore")


def train_final_model_for_window(
    df_all: pd.DataFrame,
    train_months: int,
    feature_cols: list[str],
) -> tuple[object, pd.DataFrame, str]:
    train_df = get_latest_window_df(df_all, train_months=train_months)
    X_train = np.asarray(train_df[feature_cols], dtype=np.float32)
    y_train = train_df["is_canceled"].values
    if DEFAULT_MODEL_BACKEND == "xgb":
        from xgboost import XGBClassifier

        xgb_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": 1,
        }
        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        model_name = "xgb"
    else:
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train, y_train)
        model_name = "rf"
    return model, train_df, model_name


def assign_risk_tier(row: pd.Series) -> str:
    if row["pred_prob"] >= max(row["threshold"], 0.35) and row["expected_revenue"] >= row["expected_revenue_p75"]:
        return "Tier 1 Critical"
    if row["pred_prob"] >= row["threshold"] and row["expected_revenue"] >= row["expected_revenue_p75"]:
        return "Tier 2 Revenue-sensitive"
    if row["pred_prob"] >= row["threshold"]:
        return "Tier 3 Operational risk"
    return "Tier 4 Monitor"


def assign_action(row: pd.Series) -> str:
    if row["risk_tier"] == "Tier 1 Critical":
        return "Manual review, prepayment/deposit check, overbooking reference"
    if row["risk_tier"] == "Tier 2 Revenue-sensitive":
        return "Reconfirmation and payment-status follow-up"
    if row["risk_tier"] == "Tier 3 Operational risk":
        return "Standard intervention queue"
    return "Monitor only"


def score_dataset(
    df: pd.DataFrame,
    model_bundle_map: dict[str, dict[str, object]],
    feature_cols: list[str],
) -> pd.DataFrame:
    scored_parts = []
    revenue_p75 = float(df["expected_revenue"].quantile(0.75))

    for hotel_type, bundle in model_bundle_map.items():
        subset = df[df["hotel_type"] == hotel_type].copy()
        if subset.empty:
            continue
        X = np.asarray(subset[feature_cols], dtype=np.float32)
        subset["pred_prob"] = bundle["model"].predict_proba(X)[:, 1]
        subset["threshold"] = bundle["threshold"]
        subset["predicted_intervention"] = (subset["pred_prob"] >= subset["threshold"]).astype(int)
        subset["risk_excess"] = (subset["pred_prob"] - subset["threshold"]).clip(lower=0)
        subset["opportunity_score"] = subset["risk_excess"] * subset["expected_revenue"]
        subset["expected_revenue_p75"] = revenue_p75
        subset["risk_tier"] = subset.apply(assign_risk_tier, axis=1)
        subset["decision_action"] = subset.apply(assign_action, axis=1)
        scored_parts.append(subset)

    if not scored_parts:
        return df.iloc[0:0].copy()
    scored_df = pd.concat(scored_parts, axis=0).sort_values("arrival_date").reset_index(drop=True)
    return scored_df.drop(columns=["expected_revenue_p75"])


def build_scored_dataset(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, object]:
    model_bundle_map = {}
    for hotel_type, train_months in WINDOW_BY_HOTEL.items():
        model, train_df, model_name = train_final_model_for_window(df, train_months, feature_cols)
        model_bundle_map[hotel_type] = {
            "model": model,
            "train_df": train_df,
            "threshold": THRESHOLD_BY_HOTEL[hotel_type],
            "window": train_months,
            "model_name": model_name,
        }
    scored_df = score_dataset(df, model_bundle_map, feature_cols)
    return {
        "feature_cols": feature_cols,
        "model_bundle_map": model_bundle_map,
        "scored_df": scored_df,
    }
