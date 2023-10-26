import os
import sys
from sentence_transformers import SentenceTransformer, util
from PIL import Image, UnidentifiedImageError
import faiss
import mysql.connector as mysql
from mysql.connector import Error
import yaml
from sqlalchemy.engine import make_url
from io import BytesIO
from flask import Flask, request
import threading
import collections
import simplejson
import time

# 1. DB functions
def get_db_connection(config_path):
    config = yaml.safe_load(open(config_path))
    db_config = make_url(config["db_connection_string"])
    conn = mysql.connect(host=db_config.host,
                         user=db_config.username,
                         password=db_config.password,
                         port=db_config.port,
                         database = db_config.database)
    return conn


def create_faiss_mapping_table(connection):
    try:
        if connection.is_connected():
            cursor = connection.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS faiss (
        sha256 varchar(80) not null primary key,
        faiss_id int unsigned not null,
        INDEX index_id (faiss_id),
        INDEX sha256 (sha256)
        )""")
    except Error as e:
        print("Error creating MySQL table", e)


def insert_faiss_mapping(connection, mappings):
    try:
        if connection.is_connected():
            cursor = connection.cursor()
        query = "INSERT INTO faiss (sha256, faiss_id) VALUES (%s, %s) ON DUPLICATE KEY UPDATE faiss_id=VALUES(faiss_id)"
        cursor.executemany(query, mappings)
        connection.commit()
        cursor.close()
    except Error as e:
        print("Error inserting faiss id", e)
    except:
        print("unknown error")


def get_last_insert_content_id(connection, faiss_length):
    try:
        if connection.is_connected():
            cursor = connection.cursor()
        cursor.execute('select content_id from content where sha256 in (select sha256 from faiss where faiss_id=%s)', (faiss_length-1,))
        row = cursor.fetchone()
        if row is None:
            return 0
        else:
            return row[0]
    except Error as e:
        print("Error while retrieving last_insert", e)

def get_image_list(connection, start_number):
    try:
        if connection.is_connected():
            cursor = connection.cursor()
        cursor.execute('select content_id, sha256, content, content_type from content where content_id >= %s order by content_id limit 0, 100',(start_number,))
        rows = cursor.fetchall()
        return rows
    except Error as e:
        print("Error while retrieving image list", e)


def get_shas_by_faiss_id(connection, faiss_ids):
    try:
        if connection.is_connected():
            cursor = connection.cursor()
        formatted_ids = ", ".join([str(v) for v in faiss_ids])
        cursor.execute('select sha256, faiss_id from faiss where faiss_id in ({li}) order by FIELD(faiss_id, {li})'.format(li=formatted_ids))
        rows = cursor.fetchall()
        return rows
    except Error as e:
        print("Error while retrieving last_insert", e)


# 2. Index functions
def get_index(d):
    try:
        index = faiss.read_index("index.bin")
    except RuntimeError as e:
        print("Couldn't find index, creating new one.", e)
        index = faiss.IndexFlatIP(d)

    return index


def add_embeddings(index, model, db_conn, rows):
    image_list = []
    id_list = []
    next_faiss_id = index.ntotal
    for row in rows:
        id = row[0]
        sha256 = row[1]
        binary_content = row[2]
        content_type = row[3]
        if "image/" in content_type:
            faiss_id = next_faiss_id
            image_stream = BytesIO(binary_content)
            try:
                image_list.append(Image.open(image_stream))
            except UnidentifiedImageError as e:
                print("Couldn't open sha256: " + sha256 + ". Not a valid image")
                continue
            except:
                e = sys.exc_info()[0]
                print(e)
                continue
            id_list.append((sha256, faiss_id))
            next_faiss_id += 1

    if len(image_list) > 0:
        try:
            img_emb = model.encode(image_list)
            index.add(img_emb)
            insert_faiss_mapping(db_conn, id_list)
        except TypeError as e:
            print(str(e) + " - trying one at a time")
            add_embeddings_single(index, model, db_conn, rows)
        except ValueError as e:
            print(str(e) + " - trying one at a time")
            add_embeddings_single(index, model, db_conn, rows)


def add_embeddings_single(index, model, db_conn, rows):
    next_faiss_id = index.ntotal
    for row in rows:
        id = row[0]
        sha256 = row[1]
        binary_content = row[2]
        content_type = row[3]
        if "image/" in content_type:
            faiss_id = next_faiss_id
            image_stream = BytesIO(binary_content)
            try:
                img_emb = model.encode([Image.open(image_stream)])
                index.add(img_emb)
                insert_faiss_mapping(db_conn, [(sha256, faiss_id)])
                next_faiss_id += 1
            except UnidentifiedImageError as e:
                print("Couldn't open sha256: " + sha256 + ". Not a valid image")
                continue
            except TypeError as e:
                print("TypeError: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                continue
            except ValueError as e:
                print("ValueError: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                continue


def write_index(index):
    faiss.write_index(index, "index.bin")


def update_index(index, conn):
    last_content_id = get_last_insert_content_id(conn, index.ntotal)
    while True:
        start_content_id = last_content_id + 1
        print(start_content_id)
        rows = get_image_list(conn, start_content_id)
        if len(rows) == 0:
            print("index up to date, sleeping for 60s")
            time.sleep(60)
            continue
        add_embeddings(index, model, conn, rows)
        write_index(index)
        last_content_id = rows[-1][0]


def get_text_to_image_shas(index, conn, search_term, n=5):
    query_emb = model.encode([search_term])
    D, I = index.search(query_emb, n)
    rows = get_shas_by_faiss_id(conn, I[0])
    zipped = list(map(lambda x, y: (x[0], x[1], float(y)), rows, D[0]))
    return zipped

def get_image_to_image_shas(index, conn, image_binary, n=5):
    image_stream = BytesIO(image_binary)
    image_emb = model.encode([Image.open(image_stream)])
    D, I = index.search(image_emb, n)
    rows = get_shas_by_faiss_id(conn, I[0])
    zipped = list(map(lambda x, y: (x[0], x[1], float(y)), rows, D[0]))
    return zipped

config_path = os.getenv("DISCOVER_CONFIG_PATH")
model = SentenceTransformer('clip-ViT-B-32')
conn = get_db_connection(config_path)
create_faiss_mapping_table(conn)
index = get_index(512)

index_thread = threading.Thread(target=update_index, args=(index,conn))
index_thread.start()
print("indexing started")

# Return types
SearchResult = collections.namedtuple("SearchResult",["sha256", "faiss_id", "distance"])

# App definition
app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/search/<search_term>")
def search(search_term):
    res = get_text_to_image_shas(index, conn, search_term, 5)
    named_tuple = [SearchResult(*tuple_) for tuple_ in res]
    response = app.response_class(
        response=simplejson.dumps(named_tuple),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route("/search_by_image", methods=['POST'])
def search_by_image():
    image_binary = request.get_data()
    res = get_image_to_image_shas(index, conn, image_binary, 5)
    named_tuple = [SearchResult(*tuple_) for tuple_ in res]
    response = app.response_class(
        response=simplejson.dumps(named_tuple),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route("/ntotal")
def ntotal():
    return str(index.ntotal)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=4080, debug=True)
