import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def predict_review(text, model, tokenizer, device, max_len=128):
    model.eval()
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, padding_mask=attention_mask)
        prediction = torch.argmax(logits, dim=1).item()
        probabilities = F.softmax(logits, dim=1)
        confidence = probabilities[0][prediction].item() * 100

    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    print(f"Review: '{text}'")
    print(f"Verdict: {sentiment} (Confidence: {confidence:.1f}%)\n")


if __name__ == "__main__":
    # Example usage for interactive testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("Model inference module ready.")