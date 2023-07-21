import sys

sys.path.append("../ml_sound_demo")


from datasets import (
    Audio,
    load_dataset,
)

from ml_sound_demo import what_text

dataset = load_dataset(
    "mozilla-foundation/common_voice_13_0", "en", split="validation[:100]"
).cast_column("audio", Audio(sampling_rate=16000))

result = what_text(dataset)
print(result.text())
print(result.word_error_rate())
print(result.cosine_similarity())
