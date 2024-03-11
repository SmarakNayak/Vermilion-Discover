from sklearn.cluster import DBSCAN
import os
import datetime
from discover import Discover
import asyncio

async def run_cluster():
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    config_path = os.getenv("DISCOVER_CONFIG_PATH")
    if config_path is None:
        config_path = "ord.yaml"
    cluster = Discover()
    index = cluster.get_index(768)
    await cluster.initialize_api_pool(config_path)
    await cluster.create_dbscan_table()
    print("db initialized")
    length = index.ntotal
    print(index.ntotal)
    embeddings = index.reconstruct_n(0, length)
    print("embeddings reconstructed")
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    ordb = DBSCAN(eps=0.1, min_samples=10, metric="cosine")
    ordb.fit(embeddings)
    labels = ordb.labels_.tolist()
    print("dbscan complete")
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    shas = await cluster.get_shas_up_to_faiss_id(length)
    print("shas retreieved")
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    combined = list(map(lambda x, y: (x["sha256"], x["faiss_id"], y), shas, labels))
    print(combined)
    await cluster.insert_dbscan_class(combined)
    print("classes inserted")
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

asyncio.run(run_cluster())