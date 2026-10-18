DATASET_NAME = "dair-ai/emotion"
MODEL_NAME = "bert-base-uncased"
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
N_CLASSES = len(EMOTION_LABELS)

MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
N_EPOCHS = 7

MODEL_SAVE_PATH = "outputs/model_bert.pth"
CONFUSION_MATRIX_PATH = "outputs/confusion_matrix.png"