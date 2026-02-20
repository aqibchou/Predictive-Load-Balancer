import requests
import time

for i in range(100):
    try:
        response = requests.get("http://localhost:8000/api/test")
        print(f"Request {i+1}: {response.status_code}")
    except Exception as e:
        print(f"Request {i+1} failed: {e}")
    time.sleep(0.1)