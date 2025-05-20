import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    get_scheduler
)
from torch.optim import AdamW
import matplotlib.pyplot as plt
import json

from split_dataset import get_dataset

#################################################################################

model_name = "microsoft/deberta-v3-base"
folder = "deberta"

#################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("## Device:", device)

#################################################################################

train_dataset, val_dataset, test_dataset = get_dataset()

#################################################################################
# Токенизация
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

def tokenize_function(examples): return tokenizer(examples["text"], truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

#################################################################################
# Загрузка модели
model = DebertaV2ForSequenceClassification.from_pretrained(
    model_name, num_labels=2
).to(device)

training_args = TrainingArguments(
    output_dir=f"../tmp/results-{folder}",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=50,  # 50
    weight_decay=0.05,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none",
)

#################################################################################
# Метрики
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.tensor(logits).argmax(dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.05)

train_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=train_steps
)

#################################################################################
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    optimizers=(optimizer, lr_scheduler)
)

trainer.train()

print(f"\n🏆 Лучший чекпойнт: {trainer.state.best_model_checkpoint}")

# Сохранение модели
trainer.save_model(f"../models/{folder}-twitter-propaganda")

#################################################################################
# Логирование истории
logs = trainer.state.log_history
history = {"epochs": [], "train_loss": [], "eval_loss": [], "eval_accuracy": [], "eval_f1": []}

for log in logs:
    if "epoch" in log:
        history["epochs"].append(log["epoch"])
        history["train_loss"].append(log.get("loss", None))
        history["eval_loss"].append(log.get("eval_loss", None))
        history["eval_accuracy"].append(log.get("eval_accuracy", None))
        history["eval_f1"].append(log.get("eval_f1", None))

#################################################################################
# Финальные метрики
final_val_metrics = trainer.evaluate()
test_metrics = trainer.evaluate(test_dataset)

print(f"\n✅ Финальные метрики на валидации для {model_name}:")
print(final_val_metrics)

print(f"\n🧪 Метрики на тестовом датасете для {model_name}:")
print(test_metrics)

# Сохраняем в JSON
all_metrics = {
    "final_validation": final_val_metrics,
    "test": test_metrics,
    "history": history
}

with open(f"../results/{folder}_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

#################################################################################
# 📈 График метрик
plt.figure(figsize=(10, 5))

for metric_name, label in [
    ("eval_accuracy", "Accuracy"),
    ("eval_f1", "F1 Score"),
    ("eval_loss", "Loss"),
]:
    values = history[metric_name]
    epochs = history["epochs"]
    filtered = [(e, v) for e, v in zip(epochs, values) if v is not None]
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
