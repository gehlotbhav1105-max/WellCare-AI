"""
WellCare AI - Machine Learning Pipeline
=======================================
Trains, benchmarks, and persists multiple ML classification models:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier

Evaluates models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
Saves the best performing model pipeline to models/wellcare_model.pkl.
Provides explainable inference with class probability distributions.

DISCLAIMER:
WellCare AI is an educational wellness-support system. Its results are general
wellness indicators and should not be considered a medical diagnosis or a
substitute for professional healthcare advice.
"""

import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocessing import (
    ALL_ML_FEATURES,
    CSV_PATH,
    NUMERICAL_FEATURES,
    TARGET_VARIABLE,
    build_preprocessor,
    load_all_users,
    prepare_ml_data,
)
from scoring import calculate_wellness_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "wellcare_model.pkl")
MIN_RECORDS_FOR_ML = 10


def ensure_models_directory() -> None:
    """Ensures the models directory exists."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def train_and_evaluate_models() -> Dict[str, Any]:
    """
    Trains and compares 4 ML models on available user data.
    Selects the best model and serializes the bundle to wellcare_model.pkl.
    """
    ensure_models_directory()
    df = load_all_users()

    # Check minimum record threshold
    if len(df) < MIN_RECORDS_FOR_ML:
        return {
            "status": "insufficient_data",
            "message": (
                f"More user records are required to train a reliable machine-learning model. "
                f"Currently available: {len(df)} records (Minimum required: {MIN_RECORDS_FOR_ML})."
            ),
            "num_records": len(df),
            "comparison": [],
            "best_model_name": None,
        }

    # Ensure target has at least 2 distinct classes
    if df[TARGET_VARIABLE].nunique() < 2:
        return {
            "status": "insufficient_diversity",
            "message": (
                "Data currently contains only one wellness category. "
                "More diverse user records across multiple categories are required for classification training."
            ),
            "num_records": len(df),
            "comparison": [],
            "best_model_name": None,
        }

    X, y = prepare_ml_data(df)

    # Train / test split with stratify if multiple classes have >= 2 instances
    class_counts = y.value_counts()
    stratify_target = y if (class_counts.min() >= 2) else None

    # Determine test size (if small, test_size=0.2, else 0.25)
    test_size = 0.2 if len(df) <= 30 else 0.25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_target
    )

    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    }

    comparison_results = []
    best_f1 = -1.0
    best_pipeline = None
    best_name = None
    best_eval_metrics = {}

    labels_order = sorted(list(y.unique()))

    for name, clf in models_dict.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            cm = confusion_matrix(y_test, y_pred, labels=labels_order).tolist()

            metric_entry = {
                "Model": name,
                "Accuracy": round(acc * 100, 2),
                "Precision": round(prec * 100, 2),
                "Recall": round(rec * 100, 2),
                "F1_Score": round(f1 * 100, 2),
                "raw_f1": f1,
                "Confusion_Matrix": cm,
                "Labels": labels_order,
            }
            comparison_results.append(metric_entry)

            # Compare on F1-Score then Accuracy
            if (f1 > best_f1) or (f1 == best_f1 and acc > best_eval_metrics.get("Accuracy", 0)):
                best_f1 = f1
                best_pipeline = pipeline
                best_name = name
                best_eval_metrics = metric_entry
        except Exception as e:
            comparison_results.append({
                "Model": name,
                "Accuracy": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1_Score": 0.0,
                "raw_f1": 0.0,
                "Confusion_Matrix": [],
                "Labels": labels_order,
                "Error": str(e),
            })

    if best_pipeline is not None:
        # Fit best pipeline on entire dataset for optimal production inference
        best_pipeline.fit(X, y)

        # Extract feature importances if tree-based
        clf_step = best_pipeline.named_steps["classifier"]
        importances_dict = {}
        if hasattr(clf_step, "feature_importances_"):
            try:
                cat_encoder = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
                cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist() if hasattr(cat_encoder, "get_feature_names_out") else []
                all_feature_names = NUMERICAL_FEATURES + cat_names
                raw_importances = clf_step.feature_importances_
                if len(all_feature_names) == len(raw_importances):
                    for fname, imp in zip(all_feature_names, raw_importances):
                        importances_dict[fname] = round(float(imp) * 100, 2)
                    importances_dict = dict(sorted(importances_dict.items(), key=lambda item: item[1], reverse=True)[:10])
            except Exception:
                pass

        model_bundle = {
            "pipeline": best_pipeline,
            "best_model_name": best_name,
            "best_metrics": best_eval_metrics,
            "comparison": comparison_results,
            "classes": best_pipeline.classes_.tolist(),
            "feature_importances": importances_dict,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(df),
        }

        joblib.dump(model_bundle, MODEL_PATH)

        return {
            "status": "success",
            "message": f"Best Model '{best_name}' successfully trained and saved.",
            "num_records": len(df),
            "best_model_name": best_name,
            "best_metrics": best_eval_metrics,
            "comparison": comparison_results,
            "trained_at": model_bundle["trained_at"],
        }
    else:
        return {
            "status": "training_failed",
            "message": "Model training failed due to insufficient valid training instances.",
            "num_records": len(df),
            "comparison": comparison_results,
            "best_model_name": None,
        }


def load_model_bundle() -> Optional[Dict[str, Any]]:
    """Loads the persisted model bundle from models/wellcare_model.pkl."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


def predict_wellness(clean_user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs wellness prediction and factor analysis.
    Uses ML pipeline if available, otherwise falls back gracefully to
    the transparent scoring engine with clear user transparency.
    """
    # 1. Deterministic Scoring
    scoring_result = calculate_wellness_score(clean_user_data)
    deterministic_category = scoring_result["Wellness_Category"]
    wellness_score = scoring_result["Wellness_Score"]

    # 2. ML Prediction
    model_bundle = load_model_bundle()

    # If no model exists, attempt training
    if model_bundle is None:
        train_res = train_and_evaluate_models()
        if train_res["status"] == "success":
            model_bundle = load_model_bundle()

    predicted_category = deterministic_category
    probabilities: Dict[str, float] = {}
    confidence = 100.0
    model_name = "Rule-Based Wellness Scoring Engine"
    ml_status = "deterministic_fallback"
    ml_message = "Prediction derived directly from deterministic lifestyle scoring engine (ML model pending more training data)."

    if model_bundle is not None:
        try:
            pipeline = model_bundle["pipeline"]
            model_name = model_bundle["best_model_name"]
            
            # Format single sample DataFrame
            df_single = pd.DataFrame([clean_user_data])
            feature_cols = [c for c in ALL_ML_FEATURES if c in df_single.columns]
            X_single = df_single[feature_cols]

            pred = pipeline.predict(X_single)[0]
            predicted_category = str(pred)

            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_single)[0]
                classes = pipeline.classes_
                for cls, prob in zip(classes, probs):
                    probabilities[str(cls)] = round(float(prob) * 100, 1)
                confidence = round(float(np.max(probs)) * 100, 1)

            ml_status = "ml_trained"
            ml_message = f"Prediction successfully generated by trained {model_name} classifier."
        except Exception as e:
            ml_status = "error_fallback"
            ml_message = f"ML inference encountered error ({str(e)}). Fallback scoring applied."

    return {
        "User_ID": clean_user_data.get("User_ID", ""),
        "Name": clean_user_data.get("Name", "Anonymous"),
        "Wellness_Score": wellness_score,
        "Predicted_Category": predicted_category,
        "Deterministic_Category": deterministic_category,
        "Confidence": confidence,
        "Probabilities": probabilities,
        "Model_Name": model_name,
        "ML_Status": ml_status,
        "ML_Message": ml_message,
        "Scoring_Details": scoring_result,
        "Feature_Importances": model_bundle.get("feature_importances", {}) if model_bundle else {},
    }
