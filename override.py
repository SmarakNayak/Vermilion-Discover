import os
import csv
from discover import Discover
import asyncio

async def run_override():
    config_path = os.getenv("DISCOVER_CONFIG_PATH")
    if config_path is None:
        config_path = "ord.yaml"
    overrider = Discover()
    await overrider.initialize_api_pool(config_path)
    await overrider.initialize_update_pool(config_path)
    await overrider.create_content_moderation_table()

    with open('override.csv') as f:
        next(f)
        data = [tuple(line) for line in csv.reader(f)]
    await overrider.insert_moderation_overrides(data)

    with open('override.csv') as f:
        next(f)
        blocked_shas = []
        for line in csv.reader(f):
            sha = line[0]
            flag = line[1]
            if "BLOCKED" in flag:
                blocked_shas.append((sha,))
        await overrider.delete_blocked_content(blocked_shas)

asyncio.run(run_override())