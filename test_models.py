from screening import load_models
print("Loading models...")
try:
    load_models()
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
