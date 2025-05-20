import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    MT5Tokenizer, MT5ForConditionalGeneration
)
import numpy as np

# 🔥 Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔥 Используем устройство: {device}\n")

# Тестовые тексты
test_texts = [
    "This is a great day for our nation!",
    "Beware of fake news spread by the opposition!",
    "Join us in the rally to fight corruption.",
    "Hello, World!",
    "Thats a beutiful day!",
    "Ukraine is a great country!",
    "Russia is a great country!",
    "I love my country!",
    "I hate my country!",
    "Lets take coffee!",
]

# Модели
model_list = [
    {"name": "bert-base-multilingual-cased", "folder": "mbert", "is_seq2seq": False},
    {"name": "xlm-roberta-base", "folder": "xlmr", "is_seq2seq": False},
    {"name": "google/mt5-small", "folder": "mt5", "is_seq2seq": True},
    {"name": "microsoft/deberta-v3-base", "folder": "deberta", "is_seq2seq": False},
]

for config in model_list:
    model_name = config["name"]
    folder = config["folder"]
    is_seq2seq = config["is_seq2seq"]

    print(f"\n🔍 Проверка модели: {model_name}")

    model_path = f"../models/{folder}-twitter-propaganda"

    try:
        if is_seq2seq:
            tokenizer = MT5Tokenizer.from_pretrained(model_name)
            model = MT5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True).to(device)

            for text in test_texts:
                input_text = "Tweet classification: " + text.lower()
                inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
                outputs = model.generate(**inputs, max_length=5)
                pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Text: {text}\nPrediction: {pred}\n")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device)

            inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            for text, pred in zip(test_texts, preds):
                label = "propaganda" if pred == 1 else "not_propaganda"
                print(f"Text: {text}\nPrediction: {label}\n")

    except Exception as e:
        print(f"❌ Ошибка при проверке модели {model_name}: {e}")
