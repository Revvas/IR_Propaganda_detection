import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# === Load dataset ===
df = pd.read_csv("../data/twitter_dataset.csv")
df['label'] = df['is_propaganda'].astype(int)

# === Text cleaning ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

# === Encode labels ===
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['is_propaganda'])

# === Class distribution ===
df['label_encoded'].value_counts().plot(kind='bar', title='Class Distribution', color=['skyblue', 'salmon'])
plt.xlabel('Class (0 = not propaganda, 1 = propaganda)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("../plots/class_distribution.png")
plt.close()

# === Text length ===
df['text_length'] = df['clean_text'].apply(lambda x: len(x.split()))
plt.hist(df['text_length'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Tweet Length Distribution')
plt.xlabel('Word Count')
plt.ylabel('Number of Tweets')
plt.tight_layout()
plt.savefig("../plots/text_length_distribution.png")
plt.close()

# === Average length per class ===
avg_length_by_class = df.groupby('label_encoded')['text_length'].mean()
avg_length_by_class.plot(kind='bar', color=['blue', 'red'])
plt.title("Average Tweet Length by Class")
plt.xticks([0, 1], ['Not Propaganda', 'Propaganda'], rotation=0)
plt.ylabel("Words")
plt.tight_layout()
plt.savefig("../plots/avg_length_by_class.png")
plt.close()

# === Print summary ===
print("Sample of dataset:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nLabel encoding mapping:\n", df[['is_propaganda', 'label_encoded']].drop_duplicates())
print("\nAverage tweet length by class:\n", avg_length_by_class)
