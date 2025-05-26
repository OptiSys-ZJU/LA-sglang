import json
import asyncio
import aiohttp
import requests
import tqdm
from tqdm.asyncio import tqdm_asyncio
import tqdm.dask

async def send_one_request(session, url, req, semaphore):
    async with semaphore:
        try:
            async with session.post(url, json=req) as resp:
                text = await resp.text()
                return resp.status, text
        except Exception as e:
            return None, str(e)

async def send_async_requests_with_limit(url, reqs, max_concurrency=10):
    semaphore = asyncio.Semaphore(max_concurrency)
    results = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_one_request(session, url, req, semaphore)
            for req in reqs
        ]
        for f in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Sending Requests"):
            result = await f
            results.append(result)
    return results

def send_sync_requests_with_progress(url, reqs):
    for req in tqdm.tqdm(reqs, desc="Sending Requests"):
        try:
            response = requests.post(url, json=req)
            print(response.status_code)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    duplicate = False
    mode = ['train', 'validation']
    dataset_name = 'oasst1'
    model_name = 'DeepSeek-R1-Distill-Qwen-14B'
    async_mode = False
    max_concurrency = 20

    url = "http://localhost:30000/v1/chat/completions"
    
    if duplicate:
        file_name = f'{dataset_name}_reqs_{model_name}_duplicate.json'
    else:
        file_name = f'{dataset_name}_reqs_{model_name}_unique.json'
    with open(file_name, 'r', encoding='utf-8') as f:
        all_reqs = json.load(f)
        for m in mode:
            reqs = all_reqs[m]
            if async_mode:
                results = asyncio.run(send_async_requests_with_limit(url, reqs, max_concurrency))
            else:
                send_sync_requests_with_progress(url, reqs)
