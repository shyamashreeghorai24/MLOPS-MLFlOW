import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main():
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment('breast-cancer-rf-hp')

    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')

    # Train-test split
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Define model and parameter grid
    rf = RandomForestClassifier(random_state=random_state)
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30]
    }

    # Grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    # Parent MLflow run
    with mlflow.start_run() as parent:

        # Fit model
        grid_search.fit(X_train, y_train)

        # Log each child run
        for i in range(len(grid_search.cv_results_['params'])):
            with mlflow.start_run(nested=True):
                mlflow.log_params(grid_search.cv_results_["params"][i])
                mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])

        # Log best parameters and score to parent
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        mlflow.log_params(best_params)
        mlflow.log_metric("best_accuracy", best_score)
        mlflow.log_param("random_state", random_state)

        # Evaluate on test set
        y_pred = grid_search.best_estimator_.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Log training data
        train_df = X_train.copy()
        train_df['target'] = y_train
        try:
            train_input = mlflow.data.from_pandas(train_df)
            mlflow.log_input(train_input, "training")
        except AttributeError:
            train_df.to_csv("train.csv", index=False)
            mlflow.log_artifact("train.csv", artifact_path="data")

        # Log test data
        test_df = X_test.copy()
        test_df['target'] = y_test
        try:
            test_input = mlflow.data.from_pandas(test_df)
            mlflow.log_input(test_input, "testing")
        except AttributeError:
            test_df.to_csv("test.csv", index=False)
            mlflow.log_artifact("test.csv", artifact_path="data")

        # Log source code if available
        if "__file__" in globals():
            mlflow.log_artifact(__file__)
        else:
            print("Skipping source code logging – __file__ not defined.")

        # Log model
        mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest")

        # Set metadata
        mlflow.set_tag("author", "Shyamashree Ghorai")

        # Print best results
        print("Best Parameters:", best_params)
        print("Best CV Accuracy:", best_score)
        print("Test Accuracy:", test_accuracy)

if __name__ == "__main__":
    main()
