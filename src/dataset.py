from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader


def get_dataloaders(tokenizer, samples_per_class=30000, max_len=128, batch_size=32):
    """
    Downloads, balances, tokenizes, and returns train & test DataLoaders.
    """
    dataset = load_dataset("rohan2810/amazon-movies-meta-reviews-merged", split="train")
    dataset = dataset.select_columns(['rating', 'cleaned_text'])
    dataset = dataset.filter(lambda x: x['rating'] != 3)

    def convert_to_binary(example):
        example['label'] = 1 if example['rating'] > 3 else 0
        return example

    dataset = dataset.map(convert_to_binary)

    positives = dataset.filter(lambda x: x['label'] == 1)
    negatives = dataset.filter(lambda x: x['label'] == 0)

    pos_sampled = positives.shuffle(seed=42).select(range(samples_per_class))
    neg_sampled = negatives.shuffle(seed=42).select(range(samples_per_class))

    balanced_dataset = concatenate_datasets([pos_sampled, neg_sampled]).shuffle(seed=42)
    split_dataset = balanced_dataset.train_test_split(test_size=0.1, seed=42)

    def tokenize_function(examples):
        return tokenizer(
            examples["cleaned_text"],
            padding="max_length",
            truncation=True,
            max_length=max_len
        )

    tokenized_datasets = split_dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["cleaned_text", "rating"])
    tokenized_datasets.set_format("torch")

    train_loader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tokenized_datasets['test'], batch_size=batch_size, shuffle=False)

    return train_loader, test_loader