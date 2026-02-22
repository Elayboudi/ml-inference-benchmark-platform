import requests
import time
import numpy as np

MODEL_A_URL = "http://model-a:8000/health"
MODEL_B_URL = "http://model-b:8000/health"

PREDICT_A_URL = "http://model-a:8000/predict"
PREDICT_B_URL = "http://model-b:8000/predict"
N_REQUESTS = 200


def benchmark(url):
    latencies = []

    for _ in range(N_REQUESTS):
        payload = {"features": [0.1, 0.2, 0.3]}

        start = time.perf_counter()
        response = requests.post(url, json=payload)
        duration = (time.perf_counter() - start) * 1000

        if response.status_code != 200:
            raise RuntimeError("Request failed")

        latencies.append(duration)

    return np.array(latencies)

def wait_for_service(url, timeout=30):
    print(f"Waiting for {url} ...")
    start = time.time()

    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"{url} is ready.")
                return
        except Exception:
            pass

        if time.time() - start > timeout:
            raise RuntimeError(f"Timeout waiting for {url}")

        time.sleep(1)

def summarize(name, latencies):
    print(f"\n=== {name} Results ===")
    print(f"Requests: {len(latencies)}")
    print(f"Average: {latencies.mean():.2f} ms")
    print(f"Min: {latencies.min():.2f} ms")
    print(f"Max: {latencies.max():.2f} ms")
    print(f"P95: {np.percentile(latencies, 95):.2f} ms")


if __name__ == "__main__":
    print("Starting benchmark...\n")

    wait_for_service(MODEL_A_URL)
    wait_for_service(MODEL_B_URL)

    a_lat = benchmark(PREDICT_A_URL)
    b_lat = benchmark(PREDICT_B_URL)

    summarize("Model A", a_lat)
    summarize("Model B", b_lat)