import os
import sys
import mlflow

THRESHOLD = 0.85


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is not set.")

    mlflow.set_tracking_uri(tracking_uri)

    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    accuracy = run.data.metrics.get("accuracy")

    if accuracy is None:
        raise ValueError("Accuracy metric not found.")

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")

    if accuracy < THRESHOLD:
        print(f"Failed: accuracy {accuracy:.4f} is below {THRESHOLD}")
        sys.exit(1)

    print(f"Passed: accuracy {accuracy:.4f} meets {THRESHOLD}")


if __name__ == "__main__":
    main()