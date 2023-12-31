import torch
from datasets import (
    Dataset,
)
from evaluate import load
from numpy import ndarray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import TensorType, WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
# processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="english",
    task="transcribe",
)


# TODO: make dataset agnostic
def map_prediction(batch):
    audio = batch["audio"]
    input_features = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors=TensorType.PYTORCH,
    ).input_features

    batch["reference"] = processor.tokenizer._normalize(  # noqa: SLF001
        batch["sentence"],
    )
    with torch.no_grad():
        predicted_ids = model.generate(input_features)[0]

    transcription = processor.decode(predicted_ids, skip_special_tokens=True)
    batch["transcription"] = transcription
    batch["prediction"] = processor.tokenizer._normalize(transcription)  # noqa: SLF001

    return batch


class TranscribeResult:
    """Transcription results and metrics of given dataset."""

    result: Dataset

    def __init__(self, result: Dataset):
        """Constructs a TranscribeResult with the processed dataset."""
        self.result = result

    def text(self) -> list[str]:
        """Returns the predicted texts from the audios in the dataset."""
        return self.result["transcription"]

    def word_error_rate(self) -> float:
        """Returns the word error rate of the predicted texts."""
        # TODO: cache `wer`?
        return load("wer").compute(
            references=self.result["reference"],
            predictions=self.result["prediction"],
        )

    def cosine_similarity(self) -> ndarray:
        """Returns the cosine similarity of the predicted texts."""
        vectorizer = TfidfVectorizer()
        return cosine_similarity(
            vectorizer.fit_transform(self.result["reference"]),
            vectorizer.transform(self.result["prediction"]),
        )


# TODO: accept datasetiter, datasetdict, etc.
def what_text(dataset: Dataset) -> TranscribeResult:
    """Returns a TranscribeResult containing predictions and metrics from the given dataset."""
    # TODO: batch mapping
    return TranscribeResult(dataset.map(map_prediction))
