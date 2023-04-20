from datasets import load_from_disk, load_dataset

import json

import asyncio
import aiohttp
from aiolimiter import AsyncLimiter

TOKEN = "<SPAWNING API TOKEN>"
headers = {"Authorization": f"API {TOKEN}"}

OPT_IN_OUT_URLS_SCAN_MAX_CONCURRENT_REQUESTS_NUMBER = 100
OPT_IN_OUT_URLS_SCAN_MAX_REQUESTS_PER_SECOND = 50

BATCH_SIZE = 100000
CHUNK_SIZE = 1000
IMAGE_FEATURE = "URL"

OPT_IN_COUNT = 0
OPT_OUT_COUNT = 0


async def check_spawning(image_urls, semaphore, limiter):
    url = f"https://opts-api.spawningaiapi.com/api/v2/query/urls"
    async with aiohttp.ClientSession(headers=headers) as session:
        await semaphore.acquire()
        async with limiter:
            async with session.post(
                url=url,
                data="\n".join(image_urls).encode("utf-8")
            ) as resp:
                content = await resp.read()
                semaphore.release()
                return json.loads(content)


async def opt_in_out_task(data_items) -> (int, int, int):
    tasks = []

    semaphore = asyncio.Semaphore(value=OPT_IN_OUT_URLS_SCAN_MAX_CONCURRENT_REQUESTS_NUMBER)
    limiter = AsyncLimiter(OPT_IN_OUT_URLS_SCAN_MAX_REQUESTS_PER_SECOND, time_period=1)

    shards = [data_items["URL"][i:i + CHUNK_SIZE] for i in range(0, len(data_items["URL"]), CHUNK_SIZE)]

    for shard in shards:
        tasks.append(asyncio.create_task(check_spawning(shard, semaphore, limiter)))
    await asyncio.wait(tasks)

    content = [url for task in tasks for url in task.result()["urls"]]

    opt_in = [x["optIn"] for x in content]
    opt_out = [x["optOut"] for x in content]

    return {"OPT_IN": opt_in, "OPT_OUT": opt_out}


def async_mapping(data_items):
    global OPT_IN_COUNT, OPT_OUT_COUNT

    results = asyncio.run(opt_in_out_task(data_items))

    OPT_IN_COUNT = OPT_IN_COUNT + sum(results["OPT_IN"])
    OPT_OUT_COUNT = OPT_OUT_COUNT + sum(results["OPT_OUT"])

    return results


if __name__ == "__main__":
    ds = load_from_disk("./laion2b-en-small")
    # ds = load_dataset("laion/laion2B-en", split="train", num_proc=2)
    ds_opts = ds.map(batched=True, batch_size=BATCH_SIZE, function=async_mapping, remove_columns=ds.column_names)

    with open("./results", "w") as f:
        f.write(f"OPT_IN_COUNT: {OPT_IN_COUNT}\n")
        f.write(f"OPT_OUT_COUNT: {OPT_OUT_COUNT}\n")
        f.write(f"LENGTH: {len(ds_opts)}\n")
