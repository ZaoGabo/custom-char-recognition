import sys
import time
import requests
import subprocess
import numpy as np
from pathlib import Path

# Add root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

API_URL = "http://localhost:8000"

def test_api_flow():
    print("🚀 Starting API server for testing...")
    
    # Start API in background
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        # Wait for startup
        max_retries = 20
        for i in range(max_retries):
            try:
                resp = requests.get(f"{API_URL}/health")
                if resp.status_code == 200:
                    print("✅ API is up and running!")
                    break
            except requests.ConnectionError as e:
                time.sleep(1)
                print(f"⏳ Waiting for API... ({i+1}/{max_retries}) - {e}")
        else:
            raise RuntimeError("API failed to start")
        
        # Test Health
        resp = requests.get(f"{API_URL}/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
        print("✅ Health check passed")
        
        # Test Predict
        print("🧪 Testing /predict endpoint...")
        dummy_image = [0.0] * 784
        payload = {"image": dummy_image}
        
        start_time = time.time()
        resp = requests.post(f"{API_URL}/predict", json=payload)
        latency = (time.time() - start_time) * 1000
        
        assert resp.status_code == 200
        data = resp.json()
        
        assert "character" in data
        assert "confidence" in data
        assert "top5" in data
        assert len(data["top5"]) == 5
        
        print(f"✅ Prediction successful! Latency: {latency:.2f}ms")
        print(f"   Predicted: {data['character']} (Confidence: {data['confidence']:.4f})")
        
    finally:
        print("🛑 Stopping API server...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    try:
        test_api_flow()
        print("\n🎉 All integration tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
