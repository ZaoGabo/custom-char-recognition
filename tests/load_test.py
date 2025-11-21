import time
import requests
import concurrent.futures
import numpy as np
import statistics
from typing import List

API_URL = "http://127.0.0.1:8002/predict"
NUM_REQUESTS = 1000
CONCURRENCY = 50

def send_request(i: int) -> float:
    """Sends a single request and returns latency in seconds."""
    dummy_image = [0.0] * 784
    payload = {"image": dummy_image}
    
    start_time = time.time()
    try:
        resp = requests.post(API_URL, json=payload)
        resp.raise_for_status()
        latency = time.time() - start_time
        return latency
    except Exception as e:
        print(f"Request {i} failed: {repr(e)}")
        return None

def run_load_test():
    print(f"🚀 Starting Load Test")
    print(f"   Target: {API_URL}")
    print(f"   Requests: {NUM_REQUESTS}")
    print(f"   Concurrency: {CONCURRENCY}")
    print("-" * 40)
    
    latencies: List[float] = []
    errors = 0
    
    start_total = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(send_request, i) for i in range(NUM_REQUESTS)]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                latencies.append(result)
            else:
                errors += 1
                
    total_time = time.time() - start_total
    
    # Analysis
    if not latencies:
        print("❌ All requests failed.")
        return
        
    avg_latency = statistics.mean(latencies) * 1000
    median_latency = statistics.median(latencies) * 1000
    p95_latency = statistics.quantiles(latencies, n=20)[18] * 1000 # 95th percentile
    p99_latency = statistics.quantiles(latencies, n=100)[98] * 1000 # 99th percentile
    rps = len(latencies) / total_time
    
    print("\n📊 Results:")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Successful Requests: {len(latencies)}/{NUM_REQUESTS}")
    print(f"   Errors: {errors}")
    print(f"   Throughput: {rps:.2f} req/s")
    print("-" * 20)
    print(f"   Avg Latency: {avg_latency:.2f}ms")
    print(f"   Median Latency: {median_latency:.2f}ms")
    print(f"   P95 Latency: {p95_latency:.2f}ms")
    print(f"   P99 Latency: {p99_latency:.2f}ms")
    
    if errors > 0:
        print("\n⚠️ Warning: Some requests failed.")
    else:
        print("\n✅ Load test completed successfully.")

if __name__ == "__main__":
    run_load_test()
