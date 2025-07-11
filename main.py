
import torch
import numpy as np
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
import os

# Modular imports
from llm_modules.model_utils import SimpleLLM
from llm_modules.tokenizer_utils import SimpleTokenizer, SubwordTokenizer
from llm_modules.data_utils import TextDataset, clean_text, load_conversation_dataset, fetch_all_datasets
from llm_modules.train_utils import train_model
from llm_modules.test_utils import test_model_interactively


# List of datasets to use
DATASET_CONFIGS = [
    # Conversational/Chat
    {"name": "blended_skill_talk", "split": "train", "field": "context", "type": "conversation"},
    # Instruction/QA
    {"name": "openai_humaneval", "split": "test", "field": "prompt", "type": "instruction"},
    # General Text
    {"name": "wikitext", "config": "wikitext-103-raw-v1", "split": "train", "field": "text", "type": "text"},
]

# Configurable parameters
MODEL_CONFIG = {
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 8,
    "d_ff": 2048,
    "max_seq_len": 128,
    "batch_size": 8,
    "epochs": 20,
    "use_pretrained": True,  # Set to True to use GPT-2
    "dataset_limit": 5000,
}


def create_expanded_dataset():
    """Create a more comprehensive training dataset"""
    training_texts = [
        # Conversations
        "Hello, how are you today? I am doing well, thank you for asking.",
        "What is your name? My name is Assistant, nice to meet you.",
        "How can I help you today? I would like to learn about machine learning.",
        "Thank you for your help. You are very welcome.",
        "I don't understand, can you explain? Let me clarify that for you.",
        
        # Questions and answers
        "What is artificial intelligence? AI is the simulation of human intelligence in machines.",
        "How does machine learning work? Machine learning uses algorithms to learn patterns from data.",
        "What is deep learning? Deep learning uses neural networks with multiple layers.",
        "What is programming? Programming is writing instructions for computers to follow.",
        "What is Python? Python is a popular programming language.",
        
        # Facts and knowledge
        "The sky is blue during the day. Water freezes at zero degrees Celsius.",
        "The Earth orbits around the Sun. Plants need sunlight to grow.",
        "Computers process information using binary code. The internet connects computers worldwide.",
        "Books contain knowledge and stories. Music can express emotions.",
        "Exercise is good for health. Learning new skills is rewarding.",
        
        # Simple instructions
        "To make tea, boil water and add tea leaves. Then let it steep for a few minutes.",
        "To write code, use a text editor or IDE. Start with simple programs first.",
        "To learn programming, start with basic concepts. Practice writing small programs.",
        "To solve problems, break them into smaller parts. Then solve each part step by step.",
        "To cook pasta, boil water and add salt. Then add pasta and cook until tender.",
        
        # Creative content
        "Once upon a time, there was a brave knight. The knight traveled through dark forests.",
        "In a land far away, magic was real. The wizard cast a powerful spell.",
        "The dragon guarded the treasure carefully. Many heroes tried to defeat it.",
        "The princess lived in a tall tower. She waited for someone to rescue her.",
        "The old man told stories by the fire. Children gathered around to listen.",
        
        # Technical content
        "Python is a programming language. Functions help organize code into reusable blocks.",
        "Variables store data in computer programs. Loops repeat code multiple times.",
        "Conditions help make decisions in code. Classes define objects and their behavior.",
        "Arrays store multiple values together. Strings contain text data.",
        "Databases store information efficiently. Networks connect different computers.",
        
        # Simple reasoning
        "If it rains, then the ground gets wet. Because the sun is shining, it is warm outside.",
        "Since I studied hard, I passed the test. The cat is sleeping because it is tired.",
        "When you exercise, you become stronger. Reading books increases your knowledge.",
        "If you practice music, you improve your skills. Hard work leads to success.",
        "Good food makes you healthy. Friendship makes life better.",
    ]
    
    return training_texts

# Example usage and testing
def main():
    print("=== Simple LLM Training and Testing ===\n")
    
    # Option 1: Use custom dataset
    training_texts = create_expanded_dataset()
    print(f"Training on {len(training_texts)} text examples (custom)")
    
    # Option 2: Use Hugging Face DailyDialog dataset
    use_hf_data = True  # Set to True to use DailyDialog
    
    print("Training tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=5000)
    tokenizer.train(training_texts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    if use_hf_data:
        dataset = load_conversation_dataset(tokenizer, max_length=64, limit=1000)
        print("Using DailyDialog dataset for training.")
    else:
        dataset = TextDataset(training_texts, tokenizer, max_length=64)
        print("Using custom dataset for training.")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model
    model = SimpleLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=64
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    print("\nStarting training...")
    train_model(model, dataloader, epochs=10, lr=1e-3)
    
    # Test with various prompts
    print("\n=== Testing with sample prompts ===")
    test_prompts = [
        "Hello, how are",
        "What is artificial",
        "The sky is",
        "To make tea",
        "Python is a",
        "Once upon a time",
        "I don't understand"
    ]
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt)
        if input_ids:
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            generated = model.generate(input_tensor, max_new_tokens=15, temperature=0.7)
            generated_text = tokenizer.decode(generated[0].tolist())
            print(f"Prompt: '{prompt}' -> Generated: '{generated_text}'")
    
    # Interactive testing
    try:
        test_model_interactively(model, tokenizer)
    except KeyboardInterrupt:
        print("\nTesting stopped by user.")
    
    print("\nTesting complete!")

def train_llm():
    print("=== Modular LLM Training ===\n")
    model_path = "custom_llm.pth"
    # 1. Load and clean data
    all_texts = fetch_all_datasets(None, max_length=MODEL_CONFIG["max_seq_len"], limit_per_dataset=MODEL_CONFIG["dataset_limit"])
    print("Training tokenizer...")
    # 2. Train subword tokenizer
    subword_tokenizer = SubwordTokenizer(vocab_size=5000)
    subword_tokenizer.train(all_texts)
    print("Tokenizer trained.")
    # 3. Prepare dataset
    dataset = [subword_tokenizer.encode(text)[:MODEL_CONFIG["max_seq_len"]] for text in all_texts]
    def pad_seq(seq, max_len):
        return seq + [0]*(max_len-len(seq)) if len(seq)<max_len else seq[:max_len]
    dataset = [pad_seq(seq, MODEL_CONFIG["max_seq_len"]) for seq in dataset]
    # 4. Use pretrained model if selected
    if MODEL_CONFIG["use_pretrained"]:
        print("Using pretrained GPT-2 model...")
        hf_model_dir = "hf_gpt2_model"
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if os.path.exists(hf_model_dir):
            print("Loading saved GPT-2 model...")
            model = GPT2LMHeadModel.from_pretrained(hf_model_dir)
            tokenizer = GPT2Tokenizer.from_pretrained(hf_model_dir)
        else:
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            def pad_gpt2(seq, max_len):
                seq = seq[:max_len]
                return seq + [tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else 0]*(max_len-len(seq))
            inputs = torch.tensor([
                pad_gpt2(tokenizer.encode(clean_text(text), max_length=MODEL_CONFIG["max_seq_len"], truncation=True), MODEL_CONFIG["max_seq_len"])
                for text in all_texts
            ])
            model.train()
            for epoch in range(MODEL_CONFIG["epochs"]):
                total_loss = 0
                for i in range(0, len(inputs), MODEL_CONFIG["batch_size"]):
                    batch = inputs[i:i+MODEL_CONFIG["batch_size"]]
                    batch = batch.to(model.device)
                    outputs = model(batch, labels=batch)
                    loss = outputs.loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch+1}, Loss: {total_loss/(len(inputs)//MODEL_CONFIG['batch_size']):.4f}")
            print("Training complete.")
            model.save_pretrained(hf_model_dir)
            tokenizer.save_pretrained(hf_model_dir)
            print(f"Model saved to {hf_model_dir}")
    else:
        print("Using custom transformer model...")
        model = SimpleLLM(
            vocab_size=5000,
            d_model=MODEL_CONFIG["d_model"],
            n_heads=MODEL_CONFIG["n_heads"],
            n_layers=MODEL_CONFIG["n_layers"],
            d_ff=MODEL_CONFIG["d_ff"],
            max_seq_len=MODEL_CONFIG["max_seq_len"]
        )
        if os.path.exists(model_path):
            print(f"Loading saved model from {model_path}...")
            model.load(model_path)
        else:
            dataloader = DataLoader(TextDataset([subword_tokenizer.decode(seq) for seq in dataset], subword_tokenizer, max_length=MODEL_CONFIG["max_seq_len"]), batch_size=MODEL_CONFIG["batch_size"], shuffle=True)
            train_model(model, dataloader, epochs=MODEL_CONFIG["epochs"], lr=1e-3, save_path=model_path)
    print("\n=== Testing with sample prompts ===")
    test_prompts = [
        "Hello, how are",
        "What is artificial",
        "The sky is",
        "To make tea",
        "Python is a",
        "Once upon a time",
        "I don't understand"
    ]
    for prompt in test_prompts:
        if MODEL_CONFIG["use_pretrained"]:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            output = model.generate(input_ids, max_length=MODEL_CONFIG["max_seq_len"]+20, do_sample=True)
            print(f"Prompt: '{prompt}' -> Generated: '{tokenizer.decode(output[0])}'")
        else:
            input_ids = subword_tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            generated = model.generate(input_tensor, max_new_tokens=15, temperature=0.7)
            generated_text = subword_tokenizer.decode(generated[0].tolist())
            print(f"Prompt: '{prompt}' -> Generated: '{generated_text}'")
    print("\nTesting complete!")

if __name__ == "__main__":
    train_llm()