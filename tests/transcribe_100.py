import csv
import sys
from pathlib import Path

from datasets import (
    Audio,
    load_dataset,
)

# TODO: use testing framework
sys.path.append("../ml_sound_demo")

from ml_sound_demo import what_text

dataset = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    "en",
    split="validation[:100]",
).cast_column("audio", Audio(sampling_rate=16000))

result = what_text(dataset)
print(result.text())
print(result.word_error_rate())
print(result.cosine_similarity())


with Path("output/transcription_100.txt").open("w") as output:
    output.write("\n".join(result.text()))

with Path("output/transcription_100.csv").open("w") as output:
    writer = csv.writer(output)
    for line in result.text():
        writer.writerow([line])
