import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
)

dataset = (
    load_dataset(
        "mozilla-foundation/common_voice_13_0",
        "en",
        split="train",
    )
    .select(range(10000))  # TODO: reduce gender bias
    .remove_columns(
        [
            "client_id",
            "path",
            "sentence",
            "up_votes",
            "down_votes",
            "age",
            "accent",
            "locale",
            "segment",
            "variant",
        ],
    )
    .filter(lambda x: x["gender"] != "")
    .train_test_split(test_size=0.2)
    .class_encode_column("gender")
    .rename_column("gender", "label")
)

labels = dataset["train"].features["label"].names
label2id, id2label = {}, {}
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    return feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=16000,
        truncation=True,
    )


dataset = dataset.map(preprocess_function, batched=True)

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)


num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)

training_args = TrainingArguments(
    output_dir="output/gender_classification_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train(resume_from_checkpoint=True)
trainer.evaluate()
