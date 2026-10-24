
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. Load the dataset created in Experiment 1
data = pd.read_csv("data/heatwave_data.csv")

features = [
    "max_temp",
    "min_temp",
    "humidity",
    "wind_speed",
    "previous_temp"
]

X = data[features]
y = data["heatwave"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 2. Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Heatwave Prediction")

# 3. Define the three experiments from the practical
configs = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10}
]

results = []
models = []

# 4. Run and track each configuration
for config in configs:

    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_params(config)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="heatwave_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        results.append({
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "accuracy": accuracy
        })

        models.append(model)

        print(
            f"Trees={config['n_estimators']}, "
            f"Depth={config['max_depth']}, "
            f"Accuracy={accuracy:.4f}"
        )

# 5. Compare results
results_df = pd.DataFrame(results)

print("\nExperiment comparison:")
print(results_df.to_string(index=False))

# 6. Select the model with the highest accuracy
best_index = results_df["accuracy"].idxmax()
best_model = models[best_index]

os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/heatwave_model.pkl")

print("\nBest configuration:")
print(results_df.loc[best_index].to_dict())

print("Best model saved to model/heatwave_model.pkl")
print("Experiment 2 completed.")
