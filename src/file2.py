import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub


# Initialize DagsHub tracking (this automatically sets the tracking URI)
dagshub.init(repo_owner='shyamashreeghorai24', repo_name='MLOPS-MLFlOW', mlflow=True)

# Load the dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define model parameters
max_depth = 10
n_estimators = 5

# Create or set experiment
mlflow.set_experiment("MLOPS-EXP1")

# Start MLflow run
with mlflow.start_run():
    # Train model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)

    # Log confusion matrix as artifact
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Set tags for context
    mlflow.set_tags({"Author": "Shyamashree", "Project": "Wine Classification"})

    # Log the trained model
    mlflow.sklearn.log_model(rf, "RandomForestModel")

print("✅ Model training completed successfully!")
print("📊 Model accuracy:", accuracy)
print("📁 Run logged to your DagsHub repository.")
