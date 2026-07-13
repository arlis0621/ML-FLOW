
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer
import mlflow.data
from mlflow.data.pandas_dataset import from_pandas as pandas_dataset_from_pandas
# 1. Load the dataset as a pandas DataFrame
# load_breast_cancer can return either a Bunch or (X, y) depending on return_X_y.
# Request return_X_y=True so we directly get X (DataFrame) and y (Series).
X, y = load_breast_cancer(as_frame=True, return_X_y=True)
rf=RandomForestClassifier()

param_grid={
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
    # 'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mlflow.set_experiment("Hyperparameter_Tuning_Example")
with mlflow.start_run():
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_score", best_score)
    
    train_df = X_train.copy()
    train_df['target'] = y_train
    # Convert DataFrame to an MLflow Dataset and log it
    train_dataset = pandas_dataset_from_pandas(train_df, targets='target', name="training_data")
    mlflow.log_input(train_dataset, context="training")

    # Prepare and log the testing dataset
    test_df = X_test.copy()
    test_df['target'] = y_test
    # Convert DataFrame to an MLflow Dataset and log it
    test_dataset = pandas_dataset_from_pandas(test_df, targets='target', name="testing_data")
    mlflow.log_input(test_dataset, context="testing")
    mlflow.log_artifact(__file__)
    
    mlflow.sklearn.log_model(grid_search.best_estimator_, "best_random_forest_model")   
    
    mlflow.set_tag("Author", "Your Name ")
    
    
    




# grid_search.fit(X_train, y_train)
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_
# print("Best Parameters:", best_params)
# print("Best Cross-Validation Score:", best_score)

