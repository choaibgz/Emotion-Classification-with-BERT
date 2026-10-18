from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from config import DATASET_NAME, EMOTION_LABELS


def load_emotion_dataset():
    dataset = load_dataset(DATASET_NAME)
    print(dataset)

    train_df = pd.DataFrame(dataset["train"])
    val_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])

    return train_df, val_df, test_df


def show_sample(df, n=5):
    df_display = df.copy()
    df_display["emotion"] = df_display["label"].map(lambda x: EMOTION_LABELS[x])
    print(df_display.head(n))


def show_class_distribution(df, split_name=""):
    counts = df["label"].value_counts().sort_index()
    print(f"\n--- Distribution ({split_name}) ---")
    for label_id, count in counts.items():
        print(f"{EMOTION_LABELS[label_id]:10s}: {count}")
    return counts


def plot_class_distribution(df, split_name="train"):
    counts = df["label"].value_counts().sort_index()
    labels = [EMOTION_LABELS[i] for i in counts.index]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts.values, color="steelblue")
    plt.title(f"Distribution des émotions ({split_name})")
    plt.ylabel("Nombre d'exemples")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_df, val_df, test_df = load_emotion_dataset()
    show_sample(train_df, n=5)
    show_class_distribution(train_df, "train")
    show_class_distribution(val_df, "validation")
    show_class_distribution(test_df, "test")
    plot_class_distribution(train_df, "train")