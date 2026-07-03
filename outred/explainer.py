# outred/explainer.py
# SHAP-based feature contribution calculator for flagged outliers.

import numpy as np
from typing import List, Dict, Any, Optional


def explain_outliers(
    X: np.ndarray,
    model,
    outlier_mask: np.ndarray,
    column_names: List[str],
    max_rows: int = 100,
) -> List[Dict[str, Any]]:
    """
    For each flagged outlier (up to *max_rows*), compute the top 3 features
    that contributed most to its anomaly score.

    Uses SHAP KernelExplainer which works with any model that has a
    decision_function callable.  Imported lazily so SHAP is only loaded
    when --explain is actually requested.

    Returns a list of dicts, one per explained row:
      {
        "row_index": int,
        "method": "shap" | "zscore_fallback",
        "top_features": [
          {"feature": str, "value": float, "actual_value": float,
           "median_value": float},
          ...
        ]
      }

    NOTE on "method": when SHAP fails or is unavailable, this falls back to
    a z-score-based explanation (see _fallback_explanations). The "value"
    field in that case is a z-score, NOT a SHAP value  - they are different
    statistics with different scales and interpretations. Earlier versions
    of this function labeled both as "shap_value", which silently mislabels
    fallback output as real SHAP attribution. Any caller (frontend, report,
    API consumer) MUST check "method" before interpreting "value" as a SHAP
    contribution.
    """
    if X.shape[1] == 0 or not column_names:
        return []

    outlier_indices = np.where(outlier_mask)[0]
    if len(outlier_indices) == 0:
        return []

    # Cap the number of rows we explain (SHAP is slow)
    explain_indices = outlier_indices[:max_rows]

    try:
        import shap

        # Use a background sample for KernelExplainer (100 rows max)
        bg_size = min(100, X.shape[0])
        background = X[np.random.choice(X.shape[0], bg_size, replace=False)]

        explainer = shap.KernelExplainer(model.decision_function, background)

        X_explain = X[explain_indices]
        shap_values = explainer.shap_values(X_explain, nsamples=50)
    except Exception:
        # If SHAP fails for any reason, return basic statistical explanations.
        # These are clearly labeled as a fallback, not silently treated as SHAP.
        return _fallback_explanations(X, outlier_mask, column_names, explain_indices)

    # Compute column medians for context
    col_medians = np.median(X, axis=0)

    results: List[Dict[str, Any]] = []
    for i, idx in enumerate(explain_indices):
        sv = shap_values[i] if len(shap_values.shape) > 1 else shap_values
        abs_sv = np.abs(sv)
        top_k = min(3, len(column_names))
        top_indices = np.argsort(abs_sv)[-top_k:][::-1]

        features = []
        for fi in top_indices:
            features.append({
                "feature": column_names[fi],
                "value": round(float(sv[fi]), 4),
                "actual_value": round(float(X[idx, fi]), 4),
                "median_value": round(float(col_medians[fi]), 4),
            })

        results.append({
            "row_index": int(idx),
            "method": "shap",
            "top_features": features,
        })

    return results


def _fallback_explanations(
    X: np.ndarray,
    outlier_mask: np.ndarray,
    column_names: List[str],
    explain_indices: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    When SHAP isn't available or fails, produce simple z-score-based
    explanations: which features have the largest deviation from the median?

    IMPORTANT: this returns "method": "zscore_fallback" and a "value" field
    that holds a z-score, not a SHAP value. Z-scores and SHAP values are not
    interchangeable: a z-score only measures how far a single feature
    deviates from its own column's typical range, while a SHAP value
    measures that feature's actual contribution to the model's anomaly
    score, accounting for interactions with other features. Treating one as
    the other can misrepresent why a row was flagged.
    """
    col_medians = np.median(X, axis=0)
    col_stds = np.std(X, axis=0)
    col_stds[col_stds == 0] = 1.0  # avoid division by zero

    results: List[Dict[str, Any]] = []
    for idx in explain_indices:
        row = X[idx]
        z_scores = np.abs((row - col_medians) / col_stds)
        top_k = min(3, len(column_names))
        top_indices = np.argsort(z_scores)[-top_k:][::-1]

        features = []
        for fi in top_indices:
            features.append({
                "feature": column_names[fi],
                "value": round(float(z_scores[fi]), 4),
                "actual_value": round(float(row[fi]), 4),
                "median_value": round(float(col_medians[fi]), 4),
            })

        results.append({
            "row_index": int(idx),
            "method": "zscore_fallback",
            "top_features": features,
        })

    return results