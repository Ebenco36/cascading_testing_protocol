# escalation_causal/models/model_factory.py (extended)

from typing import List, Dict, Optional, Tuple, Any
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def build_sklearn_model(model_type: str, random_state: int = 42, task: str = "classification") -> Any:
    """
    Build a scikit-learn compatible model.

    Args:
        model_type: 'logit', 'rf', 'xgb'
        random_state: Random seed
        task: 'classification' or 'regression'

    Returns:
        An unfitted estimator.
    """
    mt = model_type.lower()
    if task == "classification":
        if mt == "logit":
            return LogisticRegression(max_iter=2000, solver="lbfgs")
        if mt == "rf":
            return RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=20,
                random_state=random_state,
                n_jobs=-1,
            )
        if mt == "xgb":
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=random_state,
                eval_metric="logloss",
                verbosity=0,
            )
    else:  # regression
        if mt == "logit":
            return LinearRegression()   # fallback: linear regression for continuous
        if mt == "rf":
            return RandomForestRegressor(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=20,
                random_state=random_state,
                n_jobs=-1,
            )
        if mt == "xgb":
            import xgboost as xgb
            return xgb.XGBRegressor(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=random_state,
                verbosity=0,
            )
    raise ValueError(f"Unsupported model_type={model_type} for task={task}")