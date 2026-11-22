"""Model constructors for the baseline runs.

XGBoost is optional: when it is not installed we fall back to scikit-learn's
HistGradientBoostingClassifier, which is the same algorithm family (histogram
gradient boosting) and gives comparable numbers. `tree_name()` reports which
one actually ran, so the results log never claims XGBoost when it did not.
"""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def logreg(seed=SEED):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
    )


def tree_name():
    return "XGBoost" if HAS_XGB else "HistGradientBoosting"


def tree(pos_weight, seed=SEED):
    """pos_weight = n_genuine / n_fake in the train split (~6.5)."""
    if HAS_XGB:
        return XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=pos_weight, eval_metric="aucpr",
                             random_state=seed, n_jobs=-1)
    # class_weight="balanced" is equivalent to scale_pos_weight on this 2-class
    # problem up to a constant factor on the loss
    return HistGradientBoostingClassifier(
        max_iter=600, max_depth=6, learning_rate=0.05, l2_regularization=1.0,
        early_stopping=False, class_weight="balanced", random_state=seed)
