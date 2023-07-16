# TODO:
# transcribe() - takes in a dataset and returns text
# perf() - returns performance measurement of transcription

import torch
from datasets import Audio, load_dataset
from evaluate import load
from transformers import TensorType, WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="english",
    task="transcribe",  # audio-classification
)

dataset = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    "en",
    split="validation[:100]",
)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))


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


result = dataset.map(map_to_pred)
print(result[0]["reference"])
print(result[0]["prediction"])

wer = load("wer")
print(wer.compute(references=result["reference"], predictions=result["prediction"]))

# TODO: cosine similarity
