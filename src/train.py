"""
Training pipeline for the Real Estate ML project.

Trains classification and regression models, logs everything
to MLflow for experiment tracking.
"""

import logging

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from src.config import (
    CLASSIFICATION_TARGET,
    ELASTICNET_GRID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_STATE,
    REGRESSION_TARGET,
    RF_CLASSIFIER_GRID,
    TEST_SIZE,
    XGBOOST_CLASSIFIER_GRID,
    XGBOOST_REGRESSOR_GRID,
)
from src.data_validation import run_validation
from src.feature_engineering import run_feature_engineering
from src.preprocessing import (
    build_preprocessing_pipeline,
    cap_outliers,
    drop_correlated_features,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    """Load raw data from CSV."""
    df = pd.read_csv(path)
    logger.info("Loaded data: %d rows, %d columns.", len(df), len(df.columns))
    return df


def prepare_data(df: pd.DataFrame) -> dict:
    """
    Run the full data preparation pipeline:
    validation → feature engineering → split → outliers → preprocessing.

    Returns a dict with everything needed for training.
    """
    # Validate
    run_validation(df)

    # Feature engineering
    df_eng, tier_boundaries = run_feature_engineering(df, fit=True)

    # Separate features and targets
    X = df_eng.drop(columns=[CLASSIFICATION_TARGET, REGRESSION_TARGET])
    label_encoder = LabelEncoder()
    y_class = pd.Series(
        label_encoder.fit_transform(df_eng[CLASSIFICATION_TARGET]),
        index=df_eng.index,
    )
    y_reg = df_eng[REGRESSION_TARGET]

    # Train/test split
    X_train, X_test, y_train_clf, y_test_clf = train_test_split(
        X, y_class,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_class,
    )
    y_train_reg = y_reg.loc[X_train.index]
    y_test_reg = y_reg.loc[X_test.index]

    # Cap outliers (fit on train only)
    X_train, outlier_bounds = cap_outliers(X_train, fit=True)
    X_test, _ = cap_outliers(X_test, fit=False, bounds=outlier_bounds)

    # Drop correlated features (fit on train only)
    X_train, corr_cols = drop_correlated_features(X_train, fit=True)
    X_test, _ = drop_correlated_features(X_test, fit=False, cols_to_drop=corr_cols)

    # Build and fit preprocessing pipeline
    preprocessor, num_feats, cat_feats = build_preprocessing_pipeline(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logger.info(
        "Data prepared: train=%d, test=%d, features=%d.",
        X_train_processed.shape[0],
        X_test_processed.shape[0],
        X_train_processed.shape[1],
    )

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train_clf": y_train_clf,
        "y_test_clf": y_test_clf,
        "y_train_reg": y_train_reg,
        "y_test_reg": y_test_reg,
        "preprocessor": preprocessor,
        "tier_boundaries": tier_boundaries,
        "outlier_bounds": outlier_bounds,
        "corr_cols": corr_cols,
        "label_encoder": label_encoder,
    }


def train_classifier(
    name: str,
    model,
    param_grid: dict,
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:
    """Train a classifier with RandomizedSearchCV and log to MLflow."""
    with mlflow.start_run(run_name=f"clf_{name}"):
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=20,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1_macro",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        train_score = best_model.score(X_train, y_train)
        test_score = best_model.score(X_test, y_test)

        # Cross-validation score
        cv_scores = cross_val_score(
            best_model, X_train, y_train,
            cv=5, scoring="f1_macro",
        )

                # Log to MLflow
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.log_metric("test_accuracy", test_score)
        mlflow.log_metric("cv_f1_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_std", cv_scores.std())
        if "xgboost" in name:
            mlflow.xgboost.log_model(best_model, f"model_{name}")
        else:
            mlflow.sklearn.log_model(best_model, f"model_{name}")

        logger.info(
            "%s — Test accuracy: %.3f, CV F1: %.3f ± %.3f",
            name, test_score, cv_scores.mean(), cv_scores.std(),
        )

        return {
            "name": name,
            "model": best_model,
            "best_params": search.best_params_,
            "test_accuracy": test_score,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
        }


def train_regressor(
    name: str,
    model,
    param_grid: dict,
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:
    """Train a regressor with RandomizedSearchCV and log to MLflow."""
    with mlflow.start_run(run_name=f"reg_{name}"):
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=20,
            cv=5,
            scoring="r2",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        # Cross-validation score
        cv_scores = cross_val_score(
            best_model, X_train, y_train,
            cv=5, scoring="r2",
        )

        # Test score
        from sklearn.metrics import mean_absolute_error, r2_score
        y_pred = best_model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)

        # Log to MLflow
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("cv_r2_mean", cv_scores.mean())
        mlflow.log_metric("cv_r2_std", cv_scores.std())
        if "xgboost" in name:
            mlflow.xgboost.log_model(best_model, f"model_{name}")
        else:
            mlflow.sklearn.log_model(best_model, f"model_{name}")

        logger.info(
            "%s — Test R²: %.3f, MAE: %.0f, CV R²: %.3f ± %.3f",
            name, test_r2, test_mae, cv_scores.mean(), cv_scores.std(),
        )

        return {
            "name": name,
            "model": best_model,
            "best_params": search.best_params_,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
        }


def save_artifacts(data: dict, clf_results: list, reg_results: list):
    """Save the best models and preprocessing artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save preprocessor and boundaries
    joblib.dump(data["preprocessor"], MODELS_DIR / "preprocessor.joblib")
    joblib.dump(data["tier_boundaries"], MODELS_DIR / "tier_boundaries.joblib")
    joblib.dump(data["outlier_bounds"], MODELS_DIR / "outlier_bounds.joblib")
    joblib.dump(data["corr_cols"], MODELS_DIR / "corr_cols.joblib")
    joblib.dump(data["label_encoder"], MODELS_DIR / "label_encoder.joblib")

    # Save best classifier (highest CV F1)
    best_clf = max(clf_results, key=lambda x: x["cv_f1_mean"])
    joblib.dump(best_clf["model"], MODELS_DIR / "best_classifier.joblib")
    logger.info("Best classifier: %s (CV F1: %.3f)", best_clf["name"], best_clf["cv_f1_mean"])

    # Save best regressor (highest CV R2)
    best_reg = max(reg_results, key=lambda x: x["cv_r2_mean"])
    joblib.dump(best_reg["model"], MODELS_DIR / "best_regressor.joblib")
    logger.info("Best regressor: %s (CV R²: %.3f)", best_reg["name"], best_reg["cv_r2_mean"])


def main():
    """Run the full training pipeline."""
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load and prepare data
    df = load_data("data/raw/train.csv")
    data = prepare_data(df)

    # Train classifiers
    logger.info("=== CLASSIFICATION ===")
    clf_results = []

    clf_results.append(train_classifier(
        "random_forest",
        RandomForestClassifier(random_state=RANDOM_STATE),
        RF_CLASSIFIER_GRID,
        data["X_train"], data["y_train_clf"],
        data["X_test"], data["y_test_clf"],
    ))

    clf_results.append(train_classifier(
        "xgboost",
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss"),
        XGBOOST_CLASSIFIER_GRID,
        data["X_train"], data["y_train_clf"],
        data["X_test"], data["y_test_clf"],
    ))

    # Train regressors
    logger.info("=== REGRESSION ===")
    reg_results = []

    reg_results.append(train_regressor(
        "elasticnet",
        ElasticNet(random_state=RANDOM_STATE, max_iter=10000),
        ELASTICNET_GRID,
        data["X_train"], data["y_train_reg"],
        data["X_test"], data["y_test_reg"],
    ))

    reg_results.append(train_regressor(
        "xgboost",
        XGBRegressor(random_state=RANDOM_STATE),
        XGBOOST_REGRESSOR_GRID,
        data["X_train"], data["y_train_reg"],
        data["X_test"], data["y_test_reg"],
    ))

    # Save artifacts
    save_artifacts(data, clf_results, reg_results)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()