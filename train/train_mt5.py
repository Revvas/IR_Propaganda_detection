import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, get_scheduler, DataCollatorForSeq2Seq
from torch.optim import AdamW
import matplotlib.pyplot as plt
import json
from split_dataset import get_dataset

model_name = "google/mt5-small"
folder = "mt5"

# 🔥 Используем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\U0001F525 Используем устройство:", device)

# Загрузка данных
train_dataset, val_dataset, test_dataset = get_dataset()
print(f"\U0001F4CA Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# Преобразуем label в текст для seq2seq задачи
label_map = {1: "propaganda", 0: "not_propaganda"}
train_dataset = train_dataset.map(lambda x: {"input_text": "Tweet classification: " + x["text"], "target_text": label_map[x["label"]]}, remove_columns=["text", "label"])
val_dataset = val_dataset.map(lambda x: {"input_text": "Tweet classification: " + x["text"], "target_text": label_map[x["label"]]}, remove_columns=["text", "label"])
test_dataset = test_dataset.map(lambda x: {"input_text": "Tweet classification: " + x["text"], "target_text": label_map[x["label"]]}, remove_columns=["text", "label"])

# Токенизация
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)

def preprocess_function(examples):
    inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=3)
    inputs["labels"] = targets["input_ids"]
    return inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Аргументы обучения
training_args = Seq2SeqTrainingArguments(
    output_dir=f"../tmp/results-{folder}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=50,  # 50
    weight_decay=0.05,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    predict_with_generate=True,
    report_to="none"
)

# Метрики
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pred_bin = [1 if p.strip() == "propaganda" else 0 for p in decoded_preds]
    label_bin = [1 if l.strip() == "propaganda" else 0 for l in decoded_labels]

    precision, recall, f1, _ = precision_recall_fscore_support(label_bin, pred_bin, average="binary")
    acc = accuracy_score(label_bin, pred_bin)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Оптимизатор и шедулер
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
train_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=train_steps)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    optimizers=(optimizer, lr_scheduler)
)

# Обучение
trainer.train()
print(f"\n🏆 Лучший чекпойнт: {trainer.state.best_model_checkpoint}")

# Сохранение
trainer.save_model(f"../models/{folder}-twitter-propaganda")

# Метрики
logs = trainer.state.log_history
history = {"epochs": [], "train_loss": [], "eval_loss": [], "eval_accuracy": [], "eval_f1": []}
for log in logs:
    if "epoch" in log:
        history["epochs"].append(log["epoch"])
        history["train_loss"].append(log.get("loss"))
        history["eval_loss"].append(log.get("eval_loss"))
        history["eval_accuracy"].append(log.get("eval_accuracy"))
        history["eval_f1"].append(log.get("eval_f1"))

final_val_metrics = trainer.evaluate()
test_metrics = trainer.evaluate(test_dataset)

print("\n✅ Финальные метрики на валидации:", final_val_metrics)
print("\n🧪 Метрики на тесте:", test_metrics)

all_metrics = {
    "final_validation": final_val_metrics,
    "test": test_metrics,
    "history": history
}
with open(f"../results/{folder}_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

# 📈 График
plt.figure(figsize=(10, 5))
for metric_name, label in [
    ("eval_accuracy", "Accuracy"),
    ("eval_f1", "F1 Score"),
    ("eval_loss", "Loss")
]:
    values = history[metric_name]
    filtered = [(e, v) for e, v in zip(history["epochs"], values) if v is not None]
    if filtered:
        x, y = zip(*filtered)
        plt.plot(x, y, label=label)

plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title(f"📊 Metrics for {model_name}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"../plots/{folder}_metrics.png")
plt.close()
