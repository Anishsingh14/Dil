import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, recall_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")


def load_and_preprocess_data(csv_path: str):
    df = pd.read_csv(csv_path)
    feature_cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.int32)
    return X, y, feature_cols


def train_ensemble_model(
    csv_path: str = "dil_heart_disease_uci_cleveland.csv",
    model_path: str = "models/tabular_ensemble.pkl",
    scaler_path: str = "models/scaler.joblib",
    metadata_path: str = "models/tabular_metadata.json",
    test_size: float = 0.2,
    random_state: int = 42,
):
    X, y, feature_cols = load_and_preprocess_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class balance: pos_weight = {scale_pos_weight:.3f}")

    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        tree_method="hist",
        max_depth=5,
        learning_rate=0.03,
        n_estimators=800,
        subsample=0.85,
        colsample_bytree=0.85,
        colsample_bylevel=0.85,
        min_child_weight=2,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=0.5,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=32,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    xgb_calibrated = CalibratedClassifierCV(xgb_model, method="isotonic", cv=cv, ensemble=True)
    rf_calibrated = CalibratedClassifierCV(rf_model, method="isotonic", cv=cv, ensemble=True)
    mlp_calibrated = CalibratedClassifierCV(mlp_model, method="isotonic", cv=cv, ensemble=True)

    print("Training calibrated XGBoost...")
    xgb_calibrated.fit(X_train_scaled, y_train)
    
    print("Training calibrated Random Forest...")
    rf_calibrated.fit(X_train_scaled, y_train)
    
    print("Training calibrated MLP...")
    mlp_calibrated.fit(X_train_scaled, y_train)

    ensemble = VotingClassifier(
        estimators=[
            ("xgb", xgb_calibrated),
            ("rf", rf_calibrated),
            ("mlp", mlp_calibrated),
        ],
        voting="soft",
        weights=[2, 1, 1],
    )

    print("Training ensemble...")
    ensemble.fit(X_train_scaled, y_train)

    train_preds = ensemble.predict(X_train_scaled)
    test_preds = ensemble.predict(X_test_scaled)
    train_probs = ensemble.predict_proba(X_train_scaled)[:, 1]
    test_probs = ensemble.predict_proba(X_test_scaled)[:, 1]

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    train_recall = recall_score(y_train, train_preds)
    test_recall = recall_score(y_test, test_preds)
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)

    print(f"\n{'='*60}")
    print(f"Training Accuracy: {train_acc:.4f}, Recall: {train_recall:.4f}, AUC: {train_auc:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}, Recall: {test_recall:.4f}, AUC: {test_auc:.4f}")
    print(f"\nTest Classification Report:")
    print(classification_report(y_test, test_preds, target_names=["No Disease", "Disease"]))

    joblib.dump(ensemble, model_path)
    print(f"\nEnsemble model saved to {model_path}")

    xgb_base = xgb_calibrated.calibrated_classifiers_[0].estimator
    feature_importance = {}
    for i, col in enumerate(feature_cols):
        feature_importance[col] = float(xgb_base.feature_importances_[i])
    print(f"\nFeature Importance (from XGBoost):")
    for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v:.4f}")

    metadata = {
        "model_type": "Voting Ensemble (XGBoost + RF + MLP) + Isotonic Calibration",
        "model_version": "tabular-ensemble-v1.0.0",
        "feature_cols": feature_cols,
        "feature_importance": feature_importance,
        "test_accuracy": float(test_acc),
        "test_recall": float(test_recall),
        "test_auc": float(test_auc),
        "train_accuracy": float(train_acc),
        "train_recall": float(train_recall),
        "train_auc": float(train_auc),
        "scale_pos_weight": float(scale_pos_weight),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return ensemble, scaler, metadata


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    train_ensemble_model()