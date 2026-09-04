import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

from src.dataset import get_dataloaders
from src.model import TransformerEncoder, SentimentClassifier

# Configuration & Hyperparameters
TOKENIZER_NAME = "bert-base-uncased"
MAX_SEQ_LEN = 128
EMB_SIZE = 256
NUM_HEADS = 8
FF_HIDDEN_SIZE = 512
NUM_BLOCKS = 3
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EPOCHS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print("Loading and preparing datasets...")
    train_loader, test_loader = get_dataloaders(
        tokenizer=tokenizer,
        samples_per_class=30000,
        max_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE
    )

    print("Initializing Custom Transformer Encoder...")
    encoder = TransformerEncoder(
        N=NUM_BLOCKS,
        max_seq_len=MAX_SEQ_LEN,
        num_embeddings=tokenizer.vocab_size,
        emb_size=EMB_SIZE,
        att_out_size=EMB_SIZE,
        att_head_size=EMB_SIZE // NUM_HEADS,
        num_heads=NUM_HEADS,
        ff_hidden_size=FF_HIDDEN_SIZE
    )

    model = SentimentClassifier(encoder=encoder, hidden_size=EMB_SIZE, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, padding_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            if (step + 1) % 100 == 0:
                print(f"Step {step + 1:4d} | Loss: {train_loss / (step + 1):.4f} | Accuracy: {(train_correct / train_total) * 100:.2f}%")

        # Evaluation
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits = model(input_ids, padding_mask=attention_mask)
                predictions = torch.argmax(logits, dim=1)
                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)

        epoch_acc = (test_correct / test_total) * 100
        print(f"Epoch {epoch + 1} Test Accuracy: {epoch_acc:.2f}%")


if __name__ == "__main__":
    train()