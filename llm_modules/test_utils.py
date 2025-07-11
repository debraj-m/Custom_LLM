import torch

def test_model_interactively(model, tokenizer):
    import torch
    print("\n=== Interactive LLM Testing ===")
    print("Type 'quit' to exit, 'help' for options")
    print("TIP: Try prompts similar to training data for better results")
    temperature = 0.7
    max_tokens = 15
    top_k = 20
    while True:
        prompt = input("\nEnter prompt: ").strip()
        if prompt.lower() == 'quit':
            break
        elif prompt.lower() == 'help':
            print("\nCommands:")
            print("- Enter any text to generate completion")
            print("- 'quit' to exit")
            print("- 'help' for this menu")
            print("- 'temp X' to set temperature (e.g., 'temp 0.5')")
            print("- 'tokens X' to set max tokens (e.g., 'tokens 30')")
            print("- 'topk X' to set top-k filtering (e.g., 'topk 10')")
            print("- 'settings' to see current settings")
            print("\nGood prompts to try:")
            print("- 'Hello, how are'")
            print("- 'What is artificial'")
            print("- 'The sky is'")
            print("- 'Python is a'")
            print("- 'Once upon a time'")
            continue
        elif prompt.lower() == 'settings':
            print(f"Current settings:")
            print(f"- Temperature: {temperature}")
            print(f"- Max tokens: {max_tokens}")
            print(f"- Top-k: {top_k}")
            continue
        elif prompt.startswith('temp '):
            try:
                temperature = float(prompt.split()[1])
                print(f"Temperature set to {temperature}")
                continue
            except:
                print("Invalid temperature format. Use 'temp 0.8'")
                continue
        elif prompt.startswith('tokens '):
            try:
                max_tokens = int(prompt.split()[1])
                print(f"Max tokens set to {max_tokens}")
                continue
            except:
                print("Invalid tokens format. Use 'tokens 20'")
                continue
        elif prompt.startswith('topk '):
            try:
                top_k = int(prompt.split()[1])
                print(f"Top-k set to {top_k}")
                continue
            except:
                print("Invalid top-k format. Use 'topk 10'")
                continue
        if not prompt:
            continue
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            print("Could not tokenize input")
            continue
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        try:
            generated = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )
            generated_text = tokenizer.decode(generated[0].tolist())
            generated_text = generated_text.strip()
            if generated_text:
                print(f"Generated: {generated_text}")
            else:
                print("No output generated. Try a different prompt or adjust settings.")
        except Exception as e:
            print(f"Generation failed: {e}")
            print("Try adjusting temperature or using a different prompt.")
