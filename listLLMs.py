import urllib.request
import json
import sys

OLLAMA_URL = "http://localhost:11434/api/tags"

def get_ollama_models():
    try:
        with urllib.request.urlopen(OLLAMA_URL, timeout=3) as response:
            data = json.loads(response.read().decode())
            models = data.get("models", [])
            return models
    except Exception as e:
        print("Ollama not found...")
        print("Possible reasons:")
        print("1. Ollama service not started yet")
        print("2. Not availble on port 11434")
        print(f"\nError: {e}")
        return None

def main():
    print("Checking local Ollama models...\n")
    
    models = get_ollama_models()
    
    if models is None:
        sys.exit(1)
    
    if not models:
        print("Not found any installed model.")
        return
    
    print(f"Total Models Found: {len(models)}\n")
    print("-" * 50)
    
    for i, model in enumerate(models, 1):
        name = model.get("name", "Unknown")
        size = model.get("size", 0)
        size_gb = size / (1024 ** 3)
        modified = model.get("modified_at", "")[:19].replace("T", " ")
        
        print(f"{i}. {name}")
        print(f"   Size     : {size_gb:.2f} GB")
        print(f"   Modified : {modified}")
        print("-" * 50)

if __name__ == "__main__":
    main()
