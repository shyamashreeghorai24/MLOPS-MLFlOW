import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Fix: tracking URI must have correct IP address format
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ✅ load_wine() returns a Bunch object, not a function call
wine = load_wine()
X = wine.data
y = wine.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define model parameters
max_depth = 10
n_estimators = 5

# mlflow.set_experiment("MLOPS-EXP1")
# mlflow.set_experiment("MLOPS-EXP2")
with mlflow.start_run(experiment_id=283176268807719156):
    # Train model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and parameters
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    # Optional: log confusion matrix as artifact
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    #tags
    mlflow.set_tags({"Author":"Shyamashree","Project":"Wine Classification"})
    #log the model
    mlflow.sklearn.log_model(rf,"Random Forest")

print("Model accuracy:", accuracy)
