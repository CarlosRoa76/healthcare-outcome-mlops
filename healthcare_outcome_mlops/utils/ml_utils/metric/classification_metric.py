import sys
from healthcare_outcome_mlops.entity.artifact_entity import ClassificationMetricsArtifact
from healthcare_outcome_mlops.exception.exception import CustomException
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def get_classification_score(y_true, y_pred) -> ClassificationMetricsArtifact:
    try:
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        classification_metric = ClassificationMetricsArtifact(
            f1_score=f1,
            precision=precision,
            recall=recall,
            accuracy=accuracy
        )

        return classification_metric
    except Exception as e:
        raise CustomException(e, sys)