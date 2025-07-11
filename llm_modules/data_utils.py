
import re
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

def clean_text(text):
    text = re.sub(r"[\x00-\x1F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            self.examples.append(tokens)
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        if len(tokens) < self.max_length:
            tokens += [0] * (self.max_length - len(tokens))
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

def load_conversation_dataset(tokenizer, max_length=64, limit=1000):
    from datasets import load_dataset
    print("Loading DailyDialog dataset from Hugging Face...")
    dataset = load_dataset("daily_dialog", split="train")
    texts = []
    for item in dataset.select(range(min(limit, len(dataset)))):
        utterances = item["dialog"]
        text = " ".join(utterances)
        texts.append(text)
    print(f"Loaded {len(texts)} conversations.")
    return TextDataset(texts, tokenizer, max_length=max_length)

def fetch_all_datasets(tokenizer, max_length=128, limit_per_dataset=5000):
    from datasets import load_dataset
    DATASET_CONFIGS = [
        {"name": "blended_skill_talk", "split": "train", "field": "context", "type": "conversation"},
        {"name": "openai_humaneval", "split": "test", "field": "prompt", "type": "instruction"},
        {"name": "wikitext", "config": "wikitext-103-raw-v1", "split": "train", "field": "text", "type": "text"},
    ]
    all_texts = []
    for config in DATASET_CONFIGS:
        print(f"Loading {config['name']} ({config['type']})...")
        try:
            if "config" in config:
                ds = load_dataset(config["name"], config["config"], split=config["split"])
            else:
                ds = load_dataset(config["name"], split=config["split"])
            ds = ds.select(range(min(limit_per_dataset, len(ds))))
            for item in ds:
                if config["type"] == "conversation":
                    if isinstance(item[config["field"]], list):
                        text = " ".join(str(x) for x in item[config["field"]])
                    else:
                        text = str(item[config["field"]])
                elif config["type"] == "instruction":
                    text = str(item[config["field"]])
                else:
                    text = str(item[config["field"]])
                all_texts.append(clean_text(text))
            print(f"Loaded {len(ds)} samples from {config['name']}.")
        except Exception as e:
            print(f"Failed to load {config['name']}: {e}")
    print(f"Total combined samples: {len(all_texts)}")
    return all_texts
