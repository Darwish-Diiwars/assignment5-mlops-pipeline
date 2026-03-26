import os
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is not set.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("assignment5_pipeline")

    df = pd.read_csv("data/iris.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)

        run_id = run.info.run_id
        with open("model_info.txt", "w") as f:
            f.write(run_id)

        print(f"Run ID: {run_id}")
        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()