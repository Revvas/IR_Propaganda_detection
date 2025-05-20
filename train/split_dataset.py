import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
import re


def get_dataset():
    # Загрузка данных
    df = pd.read_csv("../data/twitter_dataset.csv")  # Поменяй путь, если нужно
    df['label'] = df['is_propaganda'].astype(int)

    # Очистка текста
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df['text'] = df['text'].apply(clean_text)

    # Сначала делим на 80% train и 20% temp (valid + test)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    # Теперь делим temp на 50/50 → 10% validation и 10% test
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    train_dataset = Dataset.from_dict({'text': train_texts.tolist(), 'label': train_labels.tolist()})
    val_dataset = Dataset.from_dict({'text': val_texts.tolist(), 'label': val_labels.tolist()})
    test_dataset = Dataset.from_dict({'text': test_texts.tolist(), 'label': test_labels.tolist()})

    print(f"📊 Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")


    return train_dataset, val_dataset, test_dataset



if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import Counter
    # Получаем разбиение
    train_ds, val_ds, test_ds = get_dataset()

    # Считаем количество примеров в каждом наборе по классам
    split_labels = {
        'train': [ex['label'] for ex in train_ds],
        'val': [ex['label'] for ex in val_ds],
        'test': [ex['label'] for ex in test_ds],
    }

    # Подготовка данных для DataFrame
    split_data = []
    for split, labels in split_labels.items():
        counter = Counter(labels)
        for label, count in counter.items():
            split_data.append({"Split": split, "Label": "Propaganda" if label == 1 else "Not Propaganda", "Count": count})

    split_df = pd.DataFrame(split_data)

    # Визуализация
    plt.figure(figsize=(8, 5))
    sns.barplot(data=split_df, x="Split", y="Count", hue="Label")
    plt.title("📊 Class Distribution per Dataset Split")
    plt.ylabel("Number of Examples")
    plt.xlabel("Dataset Split")
    plt.tight_layout()
    plt.savefig("../plots/dataset_split_distribution.png")
    plt.show()