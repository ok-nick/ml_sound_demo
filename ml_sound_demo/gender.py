# TODO:
# what_gender() - returns GenderResult
# GenderResult Class:
#   gender() - returns gender
#   plot_confusion_matrix() - plots confusion matrix
#   plot_roc_curve() - plots roc curve

from typing import Any

from datasets import Dataset
from transformers import (
    pipeline,
)

classifier = pipeline(
    "audio-classification",
    model="gender_classification_model/checkpoint-110",
)

# classifier = pipeline(
#     "audio-classification",
#     model="my_awesome_mind_model/checkpoint-110",
# )
# for i in range(100):
#     data = dataset["test"][i]
#     print("\nshould be: " + id2label[str(data["label"])])
#     print(data["audio"]["path"])
#     print(classifier(data["audio"]["path"]))

def map_prediction(batch):
    batch["prediction"] = classifier(batch["audio"]["path"])
    return batch


class GenderResult:
    result: Dataset

    def __init__(self, result: Dataset):
        self.result = result

    def gender(self) -> Any: # TODO: return type
        return self.result["prediction"]

    def plot_confusion_matrix(self):
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
        pass

    def plot_roc_curve(self):
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html#sklearn.metrics.RocCurveDisplay
        pass


def what_gender(dataset: Dataset) -> GenderResult:
    return GenderResult(dataset.map(map_prediction))
