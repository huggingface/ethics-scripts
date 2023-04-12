import asyncio
import aiohttp
from aiolimiter import AsyncLimiter
from datasets import load_from_disk

import math
import time
from typing import List

limiter = AsyncLimiter(1, 0.125)
TOKEN = "<SPAWNING API TOKEN>"
headers = {"Authorization": f"API {TOKEN}"}

BATCH_SIZE = 1000
IMAGE_FEATURE = "URL"  # For LAION2B-en

ds = load_from_disk("./laion2b-en-small")
num_batches = math.floor(len(ds) / BATCH_SIZE)
shards = [ds.shard(num_batches, index=i) for i in range(0, num_batches)]


async def check_spawning(image_urls: List[str], semaphore) -> bytes:
    url = f"https://opts-api.spawningaiapi.com/api/v2/query/urls"
    async with aiohttp.ClientSession(headers=headers) as session:
        await semaphore.acquire()
        async with limiter:
            # print("\n".join(image_urls))
            # print(f"Begin downloading {url} {(time.perf_counter() - s):0.4f} seconds")
            async with session.post(
                url=url,
                data="\n".join(image_urls)
            ) as resp:
                content = await resp.read()
                semaphore.release()
                return content


async def opt_in_out_task(image_urls: List[str], semaphore) -> None:
    content = await check_spawning(image_urls, semaphore)
    # print(content)
    # await upsert_to_database?
    # or maybe await ...add_to_mapped_dataset?
    # await write_to_file(pep_number, content)...


async def main() -> None:
    tasks = []
    semaphore = asyncio.Semaphore(value=100)

    for shard in shards:
        batch = shard[IMAGE_FEATURE]
        tasks.append(opt_in_out_task(batch, semaphore))
    await asyncio.wait(tasks)

if __name__ == "__main__":
    s = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - s
    print(f"Execution time: {elapsed:0.2f} seconds.")
