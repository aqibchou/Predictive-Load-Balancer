import requests
import time
from datetime import datetime

def test_prophet():
    base_url = "http://localhost:8000"
    
    # Test connection first
    print("Testing connection to load balancer")
    try:
        resp = requests.get(f"{base_url}/api/test", timeout=5)
        print(f"Connection OK (status: {resp.status_code})")
    except Exception as e:
        print(f"Connection FAILED: {e}")
        print("Make sure: docker-compose up -d")
        return
    
    pattern = [
        30,   # Hour 0: Low
        30,   # Hour 1: Low
        40,   # Hour 2: Building
        50,   # Hour 3: Building
        200,  # Hour 4: SPIKE!
        50,   # Hour 5: Drop
        40,   # Hour 6: Normal
        30,   # Hour 7: Low
        180,  # Hour 8: SPIKE!
        40,   # Hour 9: Drop
        30,   # Hour 10: Low
        30,   # Hour 11: Low
    ]
    
    print("=" * 60)
    print("PROPHET TEST - 12 MINUTES")
    print("=" * 60)
    print()
    
    for hour, traffic_rate in enumerate(pattern):
        delay = 60.0 / traffic_rate
        
        if traffic_rate >= 180:
            emoji = "SPIKE"
        elif traffic_rate >= 100:
            emoji = "HIGH"
        elif traffic_rate >= 50:
            emoji = "Medium"
        else:
            emoji = "Low"
        
        print(f"Hour {hour:2d} ({traffic_rate:3d} req/min) {emoji}", end=" ", flush=True)
        
        start = time.time()
        count = 0
        errors = 0
        
        while time.time() - start < 60:
            try:
                resp = requests.get(f"{base_url}/api/test", timeout=3)
                if resp.status_code == 200:
                    count += 1
                else:
                    errors += 1
            except requests.exceptions.Timeout:
                errors += 1
                time.sleep(0.1)  # Brief pause on timeout
            except Exception as e:
                errors += 1
                time.sleep(0.1)
            
            time.sleep(delay)
        
        actual_rate = int(count / 60 * 60)
        
        if errors > 0:
            print(f"{count} requests ({actual_rate} req/min) + {errors} errors")
        else:
            print(f"{count} requests ({actual_rate} req/min)")
    
    print()
    print("=" * 60)
    print("TEST COMPLETE - Check logs:")
    print("=" * 60)

if __name__ == "__main__":
    test_prophet()