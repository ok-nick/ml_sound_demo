import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import (
    Dataset,
)
from sklearn.feature_extraction.text import CountVectorizer
from evaluate import load
from numpy import ndarray
from sklearn.metrics.pairwise import cosine_similarity
from transformers import TensorType, WhisperForConditionalGeneration, WhisperProcessor

# processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="english",
    task="transcribe",
)


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
    result: Dataset

    def __init__(self, result: Dataset):
        self.result = result

    def text(self) -> list[str]:
        return self.result["prediction"]

    def word_error_rate(self) -> float:
        # TODO: cache `wer`?
        return load("wer").compute(
            references=self.result["reference"],
            predictions=self.result["prediction"],
        )

    def cosine_similarity(self) -> ndarray:
        vectorizer = TfidfVectorizer()
        return cosine_similarity(
            vectorizer.fit_transform(self.result["reference"]),
            vectorizer.transform(self.result["prediction"]),
        )


# TODO: accept data
def what_text(dataset: Dataset) -> TranscribeResult:
    # TODO: batch mapping
    return TranscribeResult(dataset.map(map_prediction))


# TODO: for the input wav file requirement
#       but we still need the sentence field for performance measurements..
# class Transcribe:
#     def from_file(path: str):
#         pass
#
#     def from_files(paths: list[str]):
#         pass
