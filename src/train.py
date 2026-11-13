import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)
from utils import prepare_features


def undersample_training_data(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """Balances ONLY the training split so fraud cases are not overwhelmed."""
    train_df = X_train.copy()
    train_df["Class"] = y_train.values

    fraud_df = train_df[train_df["Class"] == 1]
    legit_df = train_df[train_df["Class"] == 0].sample(
        n=len(fraud_df), random_state=random_state
    )

    balanced_df = pd.concat([fraud_df, legit_df]).sample(
        frac=1.0, random_state=random_state
    )
    return balanced_df.drop(columns=["Class"]), balanced_df["Class"]


def main():
    data_path = os.path.join("data", "creditcard.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError("data/creditcard.csv not found. Run python src/download_data.py first.")

    print("1. Loading and cleaning dataset...")
    df = pd.read_csv(data_path).dropna().drop_duplicates()
    print(f"   Total clean records: {len(df):,} | Fraud cases: {int(df['Class'].sum())} ({df['Class'].mean()*100:.3f}%)")

    # 2. Stratified Train/Test Split FIRST (Prevents Data Leakage)
    raw_train, raw_test = train_test_split(
        df, test_size=0.20, stratify=df["Class"], random_state=42
    )

    # 3. Feature Engineering & Scaling (Fit scaler ONLY on training data)
    X_train_scaled, scaler = prepare_features(raw_train, fit=True)
    y_train = raw_train["Class"]

    X_test_scaled, _ = prepare_features(raw_test, scaler=scaler, fit=False)
    y_test = raw_test["Class"]

    # 4. Random Undersampling on Training Split ONLY
    X_train_bal, y_train_bal = undersample_training_data(X_train_scaled, y_train)
    print(f"   Balanced training rows: {len(X_train_bal):,} (50% Fraud / 50% Legit)")
    print(f"   Untouched realistic test rows: {len(X_test_scaled):,} ({y_test.mean()*100:.3f}% Fraud)\n")

    # 5. Train & Benchmark 4 Models
    candidates = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=150, random_state=42),
    }

    trained_models = {}
    results = []
    test_probs = {}

    for name, model in candidates.items():
        print(f"Training {name}...")
        model.fit(X_train_bal, y_train_bal)
        trained_models[name] = model

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        test_probs[name] = y_prob

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1-Score": round(f1_score(y_test, y_pred), 4),
            "PR-AUC": round(average_precision_score(y_test, y_prob), 4),
            "TP (Caught)": int(tp),
            "FN (Missed)": int(fn),
            "FP (False Alarm)": int(fp),
        })

    metrics_df = pd.DataFrame(results).sort_values(by="PR-AUC", ascending=False)
    print("\n=== BENCHMARK RESULTS ON UNTOUCHED REALISTIC TEST SET ===")
    print(metrics_df.to_string(index=False))

    # 6. Save Artifacts for Streamlit App
    os.makedirs("models", exist_ok=True)
    best_model_name = metrics_df.iloc[0]["Model"]

    bundle = {
        "models": trained_models,
        "best_model_name": best_model_name,
        "scaler": scaler,
        "metrics_df": metrics_df,
        "y_test": y_test.values,
        "test_probs": test_probs,
        "sample_frauds": raw_test[raw_test["Class"] == 1].drop(columns=["Class"]).head(5),
        "sample_legits": raw_test[raw_test["Class"] == 0].drop(columns=["Class"]).head(5),
    }

    artifact_path = os.path.join("models", "fraud_bundle.joblib")
    joblib.dump(bundle, artifact_path)
    print(f"\nSaved best model ('{best_model_name}') and dashboard artifacts to {artifact_path}")


if __name__ == "__main__":
    main()