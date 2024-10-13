import torch
from train import O1Model, vocab, tokenize, detokenize, vocab_size

def load_model(model_path):
    # Load the state dict
    state_dict = torch.load(model_path)
    
    # Infer model parameters from the state dict
    d_model = state_dict['embed.weight'].shape[1]
    num_layers = max([int(key.split('.')[1]) for key in state_dict.keys() if key.startswith('transformer_layers.')]) + 1
    nhead = state_dict['transformer_layers.0.self_attn.in_proj_weight'].shape[0] // (3 * d_model)
    
    print(f"Inferred model parameters: d_model={d_model}, num_layers={num_layers}, nhead={nhead}")
    
    # Create the model with inferred parameters
    model = O1Model(vocab_size, d_model, nhead, num_layers)
    
    # Load the state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def chat_with_model(model):
    print("Welcome to the O1 Model Arithmetic Solver!")
    print("You can ask arithmetic questions like:")
    print("- Calculate the sum of 5 and 7")
    print("- Calculate the difference between 15 and 8")
    print("- Calculate the product of 6 and 4")
    print("- Calculate the quotient of 20 and 5")
    print("Type 'quit' to exit.")
    
    while True:
        user_input = input("\nEnter your question: ")
        if user_input.lower() == 'quit':
            break
        
        input_ids = torch.tensor([tokenize(user_input)])
        completion_tokens, reasoning_tokens, subtasks = model.generate_completion(input_ids, max_new_tokens=50)
        
        print("\nModel's thought process:")
        print("Reasoning:", detokenize(reasoning_tokens))
        print("Subtasks:")
        for i, subtask in enumerate(subtasks, 1):
            print(f"  {i}. {detokenize(subtask)}")
        
        print("\nModel's response:")
        print(detokenize(completion_tokens))

if __name__ == "__main__":
    model_path = "o1_model.pth"  # Make sure this path is correct
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully. Number of layers: {len(model.transformer_layers)}")
        chat_with_model(model)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print("Make sure you have trained the model and saved it with the correct filename.")
    except Exception as e:
        print(f"An error occurred: {e}")
