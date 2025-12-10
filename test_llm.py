#!/usr/bin/env python3
import json
import time
import requests
import statistics
from pathlib import Path

API_URL = "https://llmgeo-dev-1042524106019.us-central1.run.app/process"
TIMEOUT = 1000  # seconds

def load_test_cases(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test prompts file {path} not found.")
    with open(path, "r") as f:
        geojson = json.load(f)
    cases = []
    for feat in geojson.get("features", []):
        prop = feat.get("properties", {})
        cases.append({
            "site": prop.get("site", "unknown"),
            "task_name": prop.get("task_name", ""),
            "task": prop.get("task", "")
        })
    return cases

def send_request(task, task_name):
    payload = {"task": task, "task_name": task_name}
    start_time = time.time()
    try:
        response = requests.post(API_URL, json=payload, timeout=TIMEOUT)
        latency = time.time() - start_time
        ok = response.status_code == 200
        try:
            resp_json = response.json()
        except Exception:
            resp_json = None
        return ok, latency, resp_json, response.text
    except Exception as e:
        return False, 0, None, str(e)

def run_tests(test_cases):
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}: Site={case['site']} Task={case['task_name']} Prompt={case['task'][:50]}...")
        ok, latency, resp_json, resp_text = send_request(case["task"], case["task_name"])
        if ok and resp_json and resp_json.get("status") == "completed":
            passed = True
            error_msg = ""
        else:
            passed = False
            error_msg = resp_text[:200]
        results.append({
            "site": case["site"],
            "task_name": case["task_name"],
            "task": case["task"],
            "latency": latency,
            "http_ok": ok,
            "passed": passed,
            "error": error_msg
        })
        status = "PASS" if passed else "FAIL"
        print(f"  {status} -- latency: {latency:.2f}s -- error: {error_msg}")

    return results

def print_summary(results):
    total = len(results)
    passed = sum(r["passed"] for r in results)
    print(f"\n=== TEST SUMMARY ===")
    print(f"Total tests: {total}")
    print(f"Passed tests: {passed} ({passed/total*100:.1f}%)")
    latencies = [r["latency"] for r in results if r["http_ok"]]
    if latencies:
        print(f"Latency (s): min={min(latencies):.2f} max={max(latencies):.2f} avg={statistics.mean(latencies):.2f}")
    else:
        print("No successful requests to report latency.")

def save_results(results, filename="test_results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results to {filename}")

def main():
    test_prompts_path = "test_prompts.geojson"
    test_cases = load_test_cases(test_prompts_path)
    results = run_tests(test_cases)
    print_summary(results)
    save_results(results)

if __name__ == "__main__":
    main()
