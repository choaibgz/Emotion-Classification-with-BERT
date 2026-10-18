"""
model.py
BERT-based emotion classifier.
Owner: Mohammed
"""

import torch.nn as nn
from transformers import BertModel
from config import MODEL_NAME, N_CLASSES


class EmotionClassifier(nn.Module):
    def __init__(self, n_classes=N_CLASSES, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped = self.dropout(pooled_output)
        logits = self.classifier(dropped)
        # Raw logits returned — CrossEntropyLoss applies softmax internally.
        # Softmax is applied manually only at inference (predict.py).
        return logits