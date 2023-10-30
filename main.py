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
import waitress
import logging


class Discover:
    stop_indexer = False

    # 1. DB functions
    def initialize_db_pool(self, config_path):
        config = yaml.safe_load(open(config_path))
        db_config = make_url(config["db_connection_string"])
        conn = mysql.connect(host=db_config.host,
                             user=db_config.username,
                             password=db_config.password,
                             port=db_config.port,
                             database=db_config.database,
                             pool_name="discover_pool",
                             pool_size=25)
        return conn

    def get_connection(self):
        connection = mysql.connect(pool_name="discover_pool")
        return connection

    def get_cursor(self, connection):
        try:
            if connection.is_connected():
                cursor = connection.cursor(buffered=True)
            else:
                connection.reconnect(attempts=3, delay=0)
                cursor = connection.cursor(buffered=True)
        except:
            e = sys.exc_info()[0]
            print("Failed to get cursor" + str(e))
            return Error
        return cursor

    def create_faiss_mapping_table(self):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            cursor.execute("""CREATE TABLE IF NOT EXISTS faiss (
            sha256 varchar(80) not null primary key,
            faiss_id int unsigned not null,
            INDEX index_id (faiss_id),
            INDEX sha256 (sha256)
            )""")
            cursor.close()
            connection.close()
        except Error as e:
            print("Error creating MySQL table", e)


    def insert_faiss_mapping(self, mappings):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            query = "INSERT INTO faiss (sha256, faiss_id) VALUES (%s, %s) ON DUPLICATE KEY UPDATE faiss_id=VALUES(faiss_id)"
            cursor.executemany(query, mappings)
            connection.commit()
            cursor.close()
            connection.close()
        except Error as e:
            print("Error inserting faiss id", e)
        except:
            print("unknown error")

    def reconcile_index_with_db(self, index):
        print("Reconciling index & db..")
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            cursor.execute('select max(faiss_id) from faiss')
            row = cursor.fetchone()
            cursor.close()
            connection.close()
            if row is None or row[0] is None:
                last_faiss_id_in_db = -1
            else:
                last_faiss_id_in_db = row[0]
            last_faiss_id_in_idx = index.ntotal-1
            print("last_faiss_id_in_db: " + str(last_faiss_id_in_db) + " last_faiss_id_in_idx: " + str(last_faiss_id_in_idx))
            if last_faiss_id_in_db > last_faiss_id_in_idx:
                connection = self.get_connection()
                cursor = self.get_cursor(connection)
                cursor.execute('Delete from faiss where faiss_id>%s', (last_faiss_id_in_idx,))
                row = cursor.fetchone()
                print(row)
                cursor.close()
                connection.close()
                print("Deleted extra faiss ids from db")
            elif last_faiss_id_in_db < last_faiss_id_in_idx:
                #index.remove([i for i in range(last_faiss_id_in_db+1, last_faiss_id_in_idx)])
                print("Can't delete extra faiss entries from index, continuing as is. Consider reindexing")
            else:
                print("No reconciliation required")

            return min(last_faiss_id_in_db, last_faiss_id_in_idx)
        except Error as e:
            print("Error while reconciling faiss_ids", e)

    def get_last_insert_content_id(self):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            cursor.execute('select max(content_id) from content where sha256 in (select sha256 from faiss)')
            row = cursor.fetchone()
            cursor.close()
            connection.close()
            if row is None or row[0] is None:
                return 0
            else:
                return row[0]
        except Error as e:
            print("Error while retrieving last_insert", e)

    def get_image_list(self, start_number):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            cursor.execute('select content_id, sha256, content, content_type from content where content_id >= %s order by content_id limit 0, 100',(start_number,))
            rows = cursor.fetchall()
            cursor.close()
            connection.close()
            return rows
        except Error as e:
            print("Error while retrieving image list", e)


    def get_shas_by_faiss_id(self, faiss_ids):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            formatted_ids = ", ".join([str(v) for v in faiss_ids])
            cursor.execute('select sha256, faiss_id from faiss where faiss_id in ({li}) order by FIELD(faiss_id, {li})'.format(li=formatted_ids))
            rows = cursor.fetchall()
            cursor.close()
            connection.close()
            return rows
        except Error as e:
            print("Error while retrieving last_insert", e)

    def get_numbers_by_faiss_id(self, faiss_ids):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            formatted_ids = ", ".join([str(v) for v in faiss_ids])
            query = """with a as (select o.*, f.faiss_id from faiss f left join ordinals o on f.sha256=o.sha256 where f.faiss_id in ({li}))
                       select * from a where sequence_number in (select min(sequence_number) from a group by sha256) order by FIELD(faiss_id, {li})"""
            cursor.execute(query.format(li=formatted_ids))
            rows = cursor.fetchall()
            cursor.close()
            connection.close()
            return rows
        except Error as e:
            print("Error while retrieving numbers by faiss_id", e)

    def get_content_from_sha(self, sha256):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            cursor.execute('select sha256, content, content_type from content where sha256=\"{sha}\"'.format(sha=sha256))
            rows = cursor.fetchall()
            cursor.close()
            connection.close()
            return rows
        except Error as e:
            print("Error while retrieving last_insert", e)

    # 2. Index functions
    def get_index(self, d):
        try:
            index = faiss.read_index("index.bin")
        except RuntimeError as e:
            print("Couldn't find index, creating new one.", e)
            index = faiss.IndexFlatIP(d)

        return index


    def add_embeddings(self, index, model, rows):
        image_list = []
        id_list = []
        next_faiss_id = index.ntotal
        for row in rows:
            id = row[0]
            sha256 = row[1]
            binary_content = row[2]
            content_type = row[3]
            if "image/" in content_type and "svg" not in content_type:
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
                self.insert_faiss_mapping(id_list)
            except TypeError as e:
                print(str(e) + " - trying one at a time")
                self.add_embeddings_single(index, model, rows)
            except ValueError as e:
                print(str(e) + " - trying one at a time")
                self.add_embeddings_single(index, model, rows)
            except OSError as e:
                print(str(e) + " - trying one at a time")
                self.add_embeddings_single(index, model, rows)
            except:
                e = sys.exc_info()[0]
                print("Unknown Error: " + str(e) + " - trying one at a time")
                self.add_embeddings_single(index, model, rows)


    def add_embeddings_single(self, index, model, rows):
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
                    self.insert_faiss_mapping([(sha256, faiss_id)])
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
                except OSError as e:
                    print("OSError: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                    continue
                except:
                    e = sys.exc_info()[0]
                    print("Unknown Error: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                    continue


    def write_index(self, index):
        faiss.write_index(index, "index.bin")


    def update_index(self, index, model):
        print("Indexer starting in background..")
        self.reconcile_index_with_db(index)
        last_content_id = self.get_last_insert_content_id()
        while True:
            if self.stop_indexer:
                print("Exiting indexer")
                break
            start_content_id = last_content_id + 1
            print(start_content_id)
            rows = self.get_image_list(start_content_id)
            if rows is None:
                print("Trying again in 60s")
                time.sleep(60)
                continue
            if len(rows) == 0:
                print("index up to date, sleeping for 60s")
                time.sleep(60)
                continue
            self.add_embeddings(index, model, rows)
            self.write_index(index)
            last_content_id = rows[-1][0]
        print("Indexer exited")


    def get_text_to_image_shas(self, model, index, search_term, n=5):
        query_emb = model.encode([search_term])
        D, I = index.search(query_emb, n)
        rows = self.get_shas_by_faiss_id(I[0])
        zipped = list(map(lambda x, y: (x[0], x[1], float(y)), rows, D[0]))
        return zipped

    def get_image_to_image_shas(self, model, index, image_binary, n=5):
        image_stream = BytesIO(image_binary)
        image_emb = model.encode([Image.open(image_stream)])
        D, I = index.search(image_emb, n)
        rows = self.get_shas_by_faiss_id(I[0])
        zipped = list(map(lambda x, y: (x[0], x[1], float(y)), rows, D[0]))
        return zipped

    def get_text_to_inscription_numbers(self, model, index, search_term, n=5):
        t0 = time.time()
        query_emb = model.encode([search_term])
        D, I = index.search(query_emb, n)
        t1 = time.time()
        rows = self.get_numbers_by_faiss_id(I[0])
        zipped = list(map(lambda x, y: (x + (float(y),)), rows, D[0]))
        t2 = time.time()
        print("db: " + str(t2-t1) + ". index: " + str(t1-t0))
        return zipped

    def get_image_to_inscription_numbers(self, model, index, image_binary, n=5):
        t0 = time.time()
        image_stream = BytesIO(image_binary)
        image_emb = model.encode([Image.open(image_stream)])
        D, I = index.search(image_emb, n)
        t1 = time.time()
        rows = self.get_numbers_by_faiss_id(I[0])
        zipped = list(map(lambda x, y: (x + (float(y),)), rows, D[0]))
        t2 = time.time()
        print("db: " + str(t2-t1) + ". index: " + str(t1-t0))
        return zipped

    def get_similar_images(self, model, index, sha256, n=5):
        t0 = time.time()
        rows = self.get_content_from_sha(sha256)
        binary_content = rows[0][1]
        content_type = rows[0][2]
        if "image/" in content_type and "svg" not in content_type:
            t1 = time.time()
            image_stream = BytesIO(binary_content)
            try:
                image = Image.open(image_stream)
                image_emb = model.encode([image])
                D, I = index.search(image_emb, n)
                t2 = time.time()
                rows = self.get_numbers_by_faiss_id(I[0])
                zipped = list(map(lambda x, y: (x + (float(y),)), rows, D[0]))
                t3 = time.time()
                print("db1: " + str(t1 - t0) + ". index: " + str(t2 - t1) + ". db2: " + str(t3 - t2))
                return zipped
            except UnidentifiedImageError as e:
                print("Couldn't open sha256: " + sha256 + ". Not a valid image")
                return
            except:
                e = sys.exc_info()[0]
                print(e)
                return



# Return types
SearchResult = collections.namedtuple("SearchResult",["sha256", "faiss_id", "distance"])
FullSearchResult = collections.namedtuple("FullSearchResult",
                                          ["id",
                                           "content_length",
                                           "content_type",
                                           "genesis_fee",
                                           "genesis_height",
                                           "genesis_transaction",
                                           "pointer",
                                           "number",
                                           "sequence_number",
                                           "parent",
                                           "metaprotocol",
                                           "embedded_metadata",
                                           "sat",
                                           "timestamp",
                                           "sha256",
                                           "text",
                                           "is_json",
                                           "is_maybe_json",
                                           "is_bitmap_style",
                                           "is_recursive",
                                           "faiss_id",
                                           "distance"])


print("Creating app")
config_path = os.getenv("DISCOVER_CONFIG_PATH")
if config_path is None:
    config_path = "ord.yaml"
discover = Discover()
model = SentenceTransformer('clip-ViT-B-32')
index = discover.get_index(512)
conn = discover.initialize_db_pool(config_path)
discover.create_faiss_mapping_table()
app = Flask(__name__)

@app.route("/")
async def hello_world():
    return "Hello, World!"


@app.route("/search/<search_term>")
def search(search_term):
    n = request.args.get('n', default=9, type=int)
    rows = discover.get_text_to_inscription_numbers(model, index, search_term, min(n, 100))
    named_tuple = [FullSearchResult(*tuple_) for tuple_ in rows]
    response = app.response_class(
        response=simplejson.dumps(named_tuple),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route("/search_by_image", methods=['POST'])
def search_by_image():
    image_binary = request.get_data()
    n = request.args.get('n', default=9, type=int)
    rows = discover.get_image_to_inscription_numbers(model, index, image_binary, min(n, 100))
    named_tuple = [FullSearchResult(*tuple_) for tuple_ in rows]
    response = app.response_class(
        response=simplejson.dumps(named_tuple),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/similar/<sha256>")
def similar(sha256):
    n = request.args.get('n', default=9, type=int)
    rows = discover.get_similar_images(model, index, sha256, min(n, 100))
    named_tuple = [FullSearchResult(*tuple_) for tuple_ in rows]
    response = app.response_class(
        response=simplejson.dumps(named_tuple),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/ntotal")
def ntotal():
    return str(index.ntotal)


if __name__ == '__main__':
    print("main hit")
    index_thread = threading.Thread(target=discover.update_index, args=(index, model))
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