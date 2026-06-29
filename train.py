import pandas as pd
import warnings
import joblib
import json

warnings.filterwarnings("ignore")

from src.preprocessing import load_data, split_data
from src.pipeline import build_pipeline, get_models

from src.evaluation import (
    cross_validate_model,
    evaluate_model,
    print_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
)


# ==========================
# Load data
# ==========================
print("Loading data...")

X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)

# ==========================
# Save Coulmns
# ==========================
with open("models/features.json", "w") as f:
    json.dump(list(X.columns), f)

# ==========================
# Models
# ==========================
models = get_models()

pipelines = {}
results = {}

# ==========================
# CROSS VALIDATION ONLY
# ==========================
print("\nCross Validation Phase...\n")

for name, model in models.items():

    print(f"CV: {name}")

    pipeline = build_pipeline(model, X_train)

    cv_mean, cv_std, _ = cross_validate_model(
        pipeline,
        X_train,
        y_train,
        scoring="roc_auc",
        cv=5
    )

    pipelines[name] = pipeline

    results[name] = {
        "CV AUC": cv_mean,
        "CV Std": cv_std
    }

# ==========================
# Results Table
# ==========================
results_df = pd.DataFrame(results).T.sort_values("CV AUC", ascending=False)

print("\n================ RESULTS ================\n")
print(results_df)

# ==========================
# Best Model
# ==========================
best_model_name = results_df.index[0]
best_pipeline = pipelines[best_model_name]

print("\nBest Model:", best_model_name)

# ==========================
# FINAL TRAIN
# ==========================
print("\nTraining Best Pipeline...")

best_pipeline.fit(X_train, y_train)

# ==========================
# SAVE MODEL
# ==========================
joblib.dump(best_pipeline, "models/best_pipeline.pkl")

# ==========================
# FULL EVALUATION (NOW USING ALL FUNCTIONS)
# ==========================
print("\nEvaluating on Test Set...\n")

# 1. metrics
metrics = evaluate_model(best_pipeline, X_test, y_test)

print("\nMetrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# 2. classification report
print("\nClassification Report:")
print_classification_report(best_pipeline, X_test, y_test)


# ==========================
# SAVE RESULTS
# ==========================
results_df.to_csv("results/cv_results.csv")

with open("results/best_model.txt", "w") as f:
    f.write(best_model_name)

# ==========================
# DONE
# ==========================
print("\n==============================")
print("TRAINING COMPLETED")
print("==============================")
print("Best Model:", best_model_name)
print("Saved: models/best_pipeline.pkl")