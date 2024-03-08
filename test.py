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

print("Creating app")
config_path = os.getenv("DISCOVER_CONFIG_PATH")
if config_path is None:
    config_path = "ord.yaml"
discover = Discover()
model = SentenceTransformer('clip-ViT-L-14')
index = discover.get_index(768)

asyncio.run(discover.setup_db(config_path))
asyncio.run(discover.reconcile_index_with_db())