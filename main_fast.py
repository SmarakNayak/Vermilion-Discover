import os
import asyncio
from sentence_transformers import SentenceTransformer
import threading
from postgres import Discover
from fastapi import FastAPI, Request
import uvicorn
from contextlib import asynccontextmanager

print("Creating app")
config_path = os.getenv("DISCOVER_CONFIG_PATH")
if config_path is None:
    config_path = "ord.yaml"
discover = Discover()
model = SentenceTransformer('clip-ViT-L-14')
index = discover.get_index(768)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup database
    await discover.setup_db(config_path)
    # Start update thread
    index_thread = threading.Thread(target=asyncio.run, args=(discover.update_index(model, config_path),))
    index_thread.start()
    yield ## api server can run now
    # Stop index thread
    print("cleaning up")
    discover.stop_indexer = True
    print("FastAPI exited, waiting on index thread to finish..")
    index_thread.join()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def hello_world():
    return "Hello, World!"

@app.get("/ntotal")
def ntotal():
    return discover.index.ntotal

@app.get("/search/{search_term}")
async def search(search_term, n: int = 9):
    response = await discover.get_text_to_inscription_numbers(model, search_term, min(n, 50))
    return response

@app.post("/search_by_image")
async def search_by_image(request: Request, n: int = 9):
    image_binary = await request.body()
    response = await discover.get_image_to_inscription_numbers(model, image_binary, min(n, 50))
    return response

@app.get("/similar/{sha256}")
async def similar(sha256, n: int = 9):
    response = await discover.get_similar_images(model, sha256, min(n, 50))
    return response

@app.get("/get_class/{dbclass}")
async def get_class(dbclass, n: int = 9):
    response = await discover.get_dbclass_to_inscription_numbers(dbclass, n)
    return response

if __name__ == "__main__":
    print("main hit")
    uvicorn.run("main_fast:app", port=4080, log_level="info")