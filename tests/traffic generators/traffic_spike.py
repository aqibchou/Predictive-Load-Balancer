import requests
import time
import random

print("Generating realistic traffic with load spikess")

# Simulate traffic spike
for phase in ["normal", "spike", "cool_down"]:
    if phase == "normal":
        print("\nPhase 1: Normal traffic (20 req, slow)")
        num_requests = 20
        delay = 0.2
    elif phase == "spike":
        print("\nPhase 2: TRAFFIC SPIKE (200 req, fast)")
        num_requests = 600
        delay = 0.001  # Very fast requests
    else:
        print("\nPhase 3: Cool down (20 req, slow)")
        num_requests = 20
        delay = 0.3
    
    for i in range(num_requests):
        try:
            response = requests.get("http://localhost:8000/api/test")
            if i % 20 == 0:
                print(f"  {phase}: {i+1}/{num_requests}")
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(delay)
    
    print(f"  Waiting 30s for Q-learning to react")
    time.sleep(30)

print("Check docker ps to see if scaling occurred.")