# TODO:
# what_text() - returns TranscribeResult
# TranscribeResult Class:
#   text() - returns text
#   word_error_rate() - returns wer
#   cosine_similarity() - returns cosine similarity

import torch
from datasets import (
    Dataset,
)
from evaluate import load
from numpy import ndarray
from transformers import TensorType, WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="english",
    task="transcribe",
)

def map_to_pred(batch):
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
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html#sklearn.metrics.pairwise.cosine_similarity
        pass


def what_text(dataset: Dataset) -> TranscribeResult:
    return TranscribeResult(dataset.map(map_to_pred))
