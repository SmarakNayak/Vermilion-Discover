import os
import sys

from sentence_transformers import SentenceTransformer
from flask import Flask, request
import threading
import collections
import simplejson
import waitress
import logging
from postgres import Discover
import asyncio

# Return types
SearchResult = collections.namedtuple("SearchResult",["sha256", "faiss_id", "distance"])
FullSearchResult = collections.namedtuple("FullSearchResult",
                                          ["sequence_number",
                                           "id",
                                           "content_length",
                                           "content_type",
                                           "content_encoding",
                                           "genesis_fee",
                                           "genesis_height",
                                           "genesis_transaction",
                                           "pointer",
                                           "number",
                                           "parent",
                                           "delegate",
                                           "metaprotocol",
                                           "embedded_metadata",
                                           "sat",
                                           "charms",
                                           "timestamp",
                                           "sha256",
                                           "text",
                                           "is_json",
                                           "is_maybe_json",
                                           "is_bitmap_style",
                                           "is_recursive",
                                           "faiss_id",
                                           "distance"])
ClassResult = collections.namedtuple("ClassResult",
                                          ["sequence_number",
                                           "id",
                                           "content_length",
                                           "content_type",
                                           "content_encoding",
                                           "genesis_fee",
                                           "genesis_height",
                                           "genesis_transaction",
                                           "pointer",
                                           "number",
                                           "parent",
                                           "delegate",
                                           "metaprotocol",
                                           "embedded_metadata",
                                           "sat",
                                           "charms",
                                           "timestamp",
                                           "sha256",
                                           "text",
                                           "is_json",
                                           "is_maybe_json",
                                           "is_bitmap_style",
                                           "is_recursive"])


print("Creating app")
config_path = os.getenv("DISCOVER_CONFIG_PATH")
if config_path is None:
    config_path = "ord.yaml"
discover = Discover()
model = SentenceTransformer('clip-ViT-L-14')
index = discover.get_index(768)

asyncio.run(discover.setup_db(config_path))

app = Flask(__name__)

@app.route("/")
async def hello_world():
    return "Hello, World!"


@app.route("/search/<search_term>")
async def search(search_term):
    n = request.args.get('n', default=9, type=int)
    rows = await discover.get_text_to_inscription_numbers(model, search_term, min(n, 50))
    named_tuple = [FullSearchResult(*tuple_) for tuple_ in rows]
    response = app.response_class(
        response=simplejson.dumps(named_tuple),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route("/search_by_image", methods=['POST'])
async def search_by_image():
    image_binary = request.get_data()
    n = request.args.get('n', default=9, type=int)
    rows = await discover.get_image_to_inscription_numbers(model, image_binary, min(n, 50))
    named_tuple = [FullSearchResult(*tuple_) for tuple_ in rows]
    response = app.response_class(
        response=simplejson.dumps(named_tuple),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/similar/<sha256>")
async def similar(sha256):
    n = request.args.get('n', default=9, type=int)
    rows = await discover.get_similar_images(model, sha256, min(n, 50))
    named_tuple = [FullSearchResult(*tuple_) for tuple_ in rows]
    response = app.response_class(
        response=simplejson.dumps(named_tuple),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/ntotal")
def ntotal():
    return str(discover.index.ntotal)

@app.route("/get_class/<dbclass>")
async def get_class(dbclass):
    n = request.args.get('n', default=100, type=int)
    rows = await discover.get_dbclass_to_inscription_numbers(dbclass, n)
    named_tuple = [ClassResult(*tuple_) for tuple_ in rows]
    response = app.response_class(
        response=simplejson.dumps(named_tuple),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == '__main__':
    print("main hit")
    index_thread = threading.Thread(target=discover.update_index, args=(model,))
    index_thread.start()
    logger = logging.getLogger('waitress')
    logger.setLevel(logging.INFO)
    try:
        PORT = sys.argv[1]
    except IndexError:
        PORT = 4080
    waitress.serve(app, host="0.0.0.0", port=PORT)
    discover.stop_indexer = True
    print("Flask exited, waiting on index thread to finish..")
    index_thread.join()