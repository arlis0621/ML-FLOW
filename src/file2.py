
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
dagshub.init(repo_owner='arlis0621', repo_name='ML-FLOW', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/arlis0621/ML-FLOW.mlflow")


x, y = load_wine(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
max_depth=10
n_estimators=8
mlflow.set_experiment("MLFlow_Practice")
with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(x_train, y_train)
    y_pred=rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    cm=confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)
    
    #apply tags
    
    mlflow.set_tags({"Author": "Your Name", "Model": "Random Forest Classifier", "Dataset": "Wine Dataset"})
    
    #log the model
    mlflow.sklearn.log_model(rf, "random_forest_model")
