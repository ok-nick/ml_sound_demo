import sys

from datasets import (
    Audio,
    load_dataset,
)

# TODO: use testing framework
sys.path.append("../ml_sound_demo")

from ml_sound_demo import what_gender

dataset = (
    load_dataset(
        "mozilla-foundation/common_voice_13_0",
        "en",
        split="validation[:1000]",
    )
    .filter(lambda x: x["gender"] != "")
    # .select(range(905, 915))
    .cast_column("audio", Audio(sampling_rate=16000))
)

result = what_gender(dataset)
print(result.gender())
result.plot_confusion_matrix()
result.plot_roc_curve()
