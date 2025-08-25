"""
Simple latency benchmark (bonus). Runs multiple requests against the local app.
Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    python scripts/benchmark.py
Notes:
    - Stays with the required model for deployment.
    - For local comparison, you can set HF_MODEL env to another small model,
      but keep the deployed Space on the required model.
"""
import time
import statistics
import requests

URL = "http://127.0.0.1:8000/predict"

SAMPLES = [
    "I love this phone, battery life is great!",
    "This is the worst service I have ever experienced.",
    "Absolutely fantastic quality and fast shipping.",
    "Terrible experience, will not buy again.",
    "It works as expected."
]

def run_once():
    latencies = []
    labels = []
    for s in SAMPLES:
        t0 = time.time()
        r = requests.post(URL, json={"text": s}, timeout=30)
        t1 = time.time()
        r.raise_for_status()
        data = r.json()
        lat = (t1 - t0) * 1000.0
        latencies.append(lat)
        labels.append(data["label"])
    return latencies, labels

def main():
    # warmup
    run_once()
    # measure
    latencies, labels = run_once()
    print("Latency (ms) per request:", [round(l, 2) for l in latencies])
    print("P50:", round(statistics.median(latencies), 2))
    print("Mean:", round(statistics.mean(latencies), 2))
    print("P90:", round(sorted(latencies)[int(len(latencies)*0.9)-1], 2))
    print("Labels:", labels)

if __name__ == "__main__":
    main()