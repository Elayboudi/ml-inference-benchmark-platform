import requests
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_A_URL = "http://model-a:8000/predict"
MODEL_B_URL = "http://model-b:8000/predict"

HEALTH_A_URL = "http://model-a:8000/health"
HEALTH_B_URL = "http://model-b:8000/health"

TOTAL_REQUESTS = 1000
CONCURRENCY = 20


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


def send_request(url):
    payload = {"features": [0.1, 0.2, 0.3]}
    start = time.perf_counter()
    response = requests.post(url, json=payload)
    duration = (time.perf_counter() - start) * 1000

    if response.status_code != 200:
        raise RuntimeError("Request failed")

    return duration


def benchmark(url):
    latencies = []

    start_total = time.perf_counter()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(send_request, url) for _ in range(TOTAL_REQUESTS)]

        for future in as_completed(futures):
            latencies.append(future.result())

    total_time = time.perf_counter() - start_total
    throughput = TOTAL_REQUESTS / total_time

    return np.array(latencies), throughput


def summarize(name, latencies, throughput):
    print(f"\n=== {name} Results ===")
    print(f"Requests: {len(latencies)}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Average: {latencies.mean():.2f} ms")
    print(f"Min: {latencies.min():.2f} ms")
    print(f"Max: {latencies.max():.2f} ms")
    print(f"P95: {np.percentile(latencies, 95):.2f} ms")
    print(f"Throughput: {throughput:.2f} req/sec")


if __name__ == "__main__":
    print("Starting concurrent benchmark...\n")

    wait_for_service(HEALTH_A_URL)
    wait_for_service(HEALTH_B_URL)

    a_lat, a_tp = benchmark(MODEL_A_URL)
    b_lat, b_tp = benchmark(MODEL_B_URL)

    summarize("Model A", a_lat, a_tp)
    summarize("Model B", b_lat, b_tp)