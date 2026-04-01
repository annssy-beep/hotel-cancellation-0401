import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".cache" / "matplotlib"))

import numpy as np
import pandas as pd
import shap


CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)


def prettify_feature_name(feature_name: str) -> str:
    return feature_name.replace("_", " ")


def normalize_shap_values(shap_values):
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[0])
    arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        return arr[:, :, 1]
    return arr


def get_global_shap_summary(
    model_bundle: dict[str, object],
    subset_df: pd.DataFrame,
    feature_cols: list[str],
    top_n: int = 12,
    sample_size: int = 500,
) -> pd.DataFrame:
    if subset_df.empty:
        return pd.DataFrame()
    sample = subset_df[feature_cols]
    if len(sample) > sample_size:
        sample = sample.sample(sample_size, random_state=42)

    explainer = shap.TreeExplainer(model_bundle["model"])
    shap_values = normalize_shap_values(explainer.shap_values(sample))
    summary = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    )
    summary["direction"] = np.where(summary["mean_shap"] >= 0, "Increase risk", "Protective")
    summary["feature_label"] = summary["feature"].map(prettify_feature_name)
    return summary.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def get_local_shap_explanation(
    model_bundle: dict[str, object],
    row_df: pd.DataFrame,
    feature_cols: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    if row_df.empty:
        return pd.DataFrame()
    sample = row_df[feature_cols]
    explainer = shap.TreeExplainer(model_bundle["model"])
    shap_values = normalize_shap_values(explainer.shap_values(sample))
    row_values = shap_values[0] if shap_values.ndim == 2 else shap_values
    out = pd.DataFrame(
        {
            "feature": feature_cols,
            "feature_value": sample.iloc[0].values,
            "shap_value": row_values,
        }
    )
    out["abs_shap"] = out["shap_value"].abs()
    out["direction"] = np.where(out["shap_value"] >= 0, "Increase risk", "Protective")
    out["feature_label"] = out["feature"].map(prettify_feature_name)
    out["feature_value"] = out["feature_value"].astype(str)
    return out.sort_values("abs_shap", ascending=False).head(top_n).reset_index(drop=True)
