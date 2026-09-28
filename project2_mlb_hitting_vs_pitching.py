"""Project 2: MLB Hitting vs. Pitching — classification."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/corbtastik/lahman-baseball-db/main/Teams.csv"

df = pd.read_csv(DATA_URL)
df = df[df["yearID"].between(2000, 2025)].copy()

# Feature engineering
df["runs_per_game"] = df["R"] / df["G"]
df["home_runs_per_game"] = df["HR"] / df["G"]
df["obp"] = (df["H"] + df["BB"] + df["HBP"]) / (df["AB"] + df["BB"] + df["HBP"] + df["SF"])
df["slg"] = (df["H"] + df["2B"] + 2 * df["3B"] + 3 * df["HR"]) / df["AB"]
df["runs_allowed_per_game"] = df["RA"] / df["G"]
df["so_per_9"] = df["SOA"] / (df["IPouts"] / 3) * 9

# Target: 1 = winning season, 0 = non-winning season.
df["Winning_Season"] = (df["W"] > df["L"]).astype(int)

hitting = ["runs_per_game", "home_runs_per_game", "obp", "slg"]
pitching = ["runs_allowed_per_game", "ERA", "so_per_9"]
combined = hitting + pitching

# Do not use W, L, winning percentage, or other direct outcome variables as predictors.
model_df = df[["yearID", "teamID", "name", "Winning_Season"] + combined].dropna().copy()

print("Observations:", len(model_df))
print("Class counts:")
print(model_df["Winning_Season"].value_counts().sort_index())
print("Duplicate team-season keys:", df.duplicated(["yearID", "teamID"]).sum())

train = model_df[model_df["yearID"] <= 2022].copy()
test = model_df[model_df["yearID"] >= 2023].copy()

def metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(train[combined], train["Winning_Season"])
baseline_pred = baseline.predict(test[combined])
print("Baseline:", metrics(test["Winning_Season"], baseline_pred))

feature_sets = {"Hitting": hitting, "Pitching": pitching, "Combined": combined}
results = []
predictions = {}
models = {}

for feature_name, features in feature_sets.items():
    candidates = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    }
    for model_name, model in candidates.items():
        model.fit(train[features], train["Winning_Season"])
        pred = model.predict(test[features])
        models[(feature_name, model_name)] = model
        predictions[(feature_name, model_name)] = pred
        results.append({"Model": model_name, "Feature Set": feature_name,
                        **metrics(test["Winning_Season"], pred)})

results_df = pd.DataFrame(results)
print("\nModel results:")
print(results_df.round(3).to_string(index=False))

for key, pred in predictions.items():
    print(f"\n{key[1]} — {key[0]}")
    print(confusion_matrix(test["Winning_Season"], pred))

combined_lr = models[("Combined", "Logistic Regression")]
coef_df = pd.DataFrame({
    "Feature": combined,
    "Coefficient": combined_lr.named_steps["model"].coef_[0]
}).sort_values("Coefficient", ascending=False)
print("\nCombined Logistic Regression coefficients:")
print(coef_df.round(3).to_string(index=False))

combined_tree = models[("Combined", "Decision Tree")]
importance_df = pd.DataFrame({
    "Feature": combined,
    "Importance": combined_tree.feature_importances_
}).sort_values("Importance", ascending=False)
print("\nCombined Decision Tree feature importance:")
print(importance_df.round(3).to_string(index=False))

# Simple communication plots
results_df.pivot(index="Feature Set", columns="Model", values="Accuracy").plot(kind="bar")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Held-out Accuracy: Hitting vs. Pitching")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

importance_df.plot(x="Feature", y="Importance", kind="bar", legend=False)
plt.ylabel("Importance")
plt.title("Combined Decision Tree Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
