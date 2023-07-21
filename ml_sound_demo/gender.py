from typing import Any

from datasets import Dataset
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from transformers import (
    pipeline,
)

# TODO: can I cache this?
classifier = pipeline(
    "audio-classification",
    # TODO: customizable
    model="gender_classification_model/checkpoint-110",
)


# TODO: more idiomatic way to do this?
def gender_to_id(gender: str) -> int:
    match gender:
        case "male":
            return 0
        case "female":
            return 1
        case "other":
            return 2
        case _:
            return -1


def map_prediction(batch):
    result = classifier(batch["audio"]["path"])
    batch["result"] = result

    gender, score = "", 0
    for x in result:
        if x["score"] > score:
            score = x["score"]
            gender = x["label"]
    batch["correct"] = 1 if gender == batch["gender"] else 0
    batch["score"] = score

    batch["reference_id"] = gender_to_id(batch["gender"])
    batch["predicted_id"] = gender_to_id(gender)

    return batch


class GenderResult:
    """Gender classification results and metrics of given dataset."""

    result: Dataset

    def __init__(self, result: Dataset):
        """Constructs a GenderResult with the processed dataset."""
        self.result = result

    def gender(self) -> list[Any]:  # TODO: return type
        """Returns the predicted genders from the audios in the dataset."""
        return self.result["result"]

    def plot_confusion_matrix(self):
        """Plots a confusion matrix."""
        display = ConfusionMatrixDisplay.from_predictions(
            self.result["reference_id"],
            self.result["predicted_id"],
        )
        display.plot()
        plt.show()

    def plot_roc_curve(self):
        """Plots an ROC curve."""
        display = RocCurveDisplay.from_predictions(
            self.result["correct"],
            self.result["score"],
        )
        display.plot()
        plt.show()


def what_gender(dataset: Dataset) -> GenderResult:
    """Returns a GenderResult containing predictions and metrics from the given Dataset."""
    return GenderResult(dataset.map(map_prediction, load_from_cache_file=False))
