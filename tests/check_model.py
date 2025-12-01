import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = 'C:/Users/moham/project-group-101/models/prophet_model.pkl'

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print("Model type:", type(model))
print("\nExtra regressors:")
if hasattr(model, 'extra_regressors'):
    for name, params in model.extra_regressors.items():
        print(f"  - {name}")
else:
    print("  None")

print("\nModel attributes:")
for attr in ['seasonalities', 'growth', 'changepoints']:
    if hasattr(model, attr):
        print(f"  {attr}: {getattr(model, attr)}")