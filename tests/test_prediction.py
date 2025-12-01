import os
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['PROPHET_MODEL_PATH'] = 'C:/Users/moham/project-group-101/models/prophet_model.pkl'

from load_balancer.prediction_service import prediction_service


def main():
    print("Loading model...")
    loaded = prediction_service.load_model()
    print(f"Model loaded: {loaded}")
    
    if not loaded:
        print("Check that prophet_model.pkl exists at the path above")
        return
    
    print("\nRunning prediction...")
    asyncio.run(prediction_service._run_prediction())
    
    pred = prediction_service.get_current_prediction()
    print("\nPrediction result:")
    for key, value in pred.to_dict().items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()