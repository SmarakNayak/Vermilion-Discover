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
    def get_db_connection(self, config_path):
        config = yaml.safe_load(open(config_path))
        db_config = make_url(config["db_connection_string"])
        conn = mysql.connect(host=db_config.host,
                             user=db_config.username,
                             password=db_config.password,
                             port=db_config.port,
                             database=db_config.database)
        return conn

    def get_cursor(self, connection):
        try:
            if connection.is_connected():
                cursor = connection.cursor(buffered=True)
            else:
                connection.reconnect(attempts=3, delay=0.1)
                cursor = connection.cursor(buffered=True)
        except:
            e = sys.exc_info()[0]
            print("Failed to get cursor" + str(e))
            return Error
        return cursor

    def create_faiss_mapping_table(self, connection):
        try:
            cursor = self.get_cursor(connection)
            cursor.execute("""CREATE TABLE IF NOT EXISTS faiss (
            sha256 varchar(80) not null primary key,
            faiss_id int unsigned not null,
            INDEX index_id (faiss_id),
            INDEX sha256 (sha256)
            )""")
        except Error as e:
            print("Error creating MySQL table", e)


    def insert_faiss_mapping(self, connection, mappings):
        try:
            cursor = self.get_cursor(connection)
            query = "INSERT INTO faiss (sha256, faiss_id) VALUES (%s, %s) ON DUPLICATE KEY UPDATE faiss_id=VALUES(faiss_id)"
            cursor.executemany(query, mappings)
            connection.commit()
            cursor.close()
        except Error as e:
            print("Error inserting faiss id", e)
        except:
            print("unknown error")


    def get_last_insert_content_id(self, connection, faiss_length):
        try:
            cursor = self.get_cursor(connection)
            cursor.execute('select content_id from content where sha256 in (select sha256 from faiss where faiss_id=%s)', (faiss_length-1,))
            row = cursor.fetchone()
            if row is None:
                return 0
            else:
                return row[0]
        except Error as e:
            print("Error while retrieving last_insert", e)

    def get_image_list(self, connection, start_number):
        try:
            cursor = self.get_cursor(connection)
            cursor.execute('select content_id, sha256, content, content_type from content where content_id >= %s order by content_id limit 0, 100',(start_number,))
            rows = cursor.fetchall()
            return rows
        except Error as e:
            print("Error while retrieving image list", e)


    def get_shas_by_faiss_id(self, connection, faiss_ids):
        try:
            cursor = self.get_cursor(connection)
            formatted_ids = ", ".join([str(v) for v in faiss_ids])
            cursor.execute('select sha256, faiss_id from faiss where faiss_id in ({li}) order by FIELD(faiss_id, {li})'.format(li=formatted_ids))
            rows = cursor.fetchall()
            return rows
        except Error as e:
            print("Error while retrieving last_insert", e)

    def get_numbers_by_faiss_id(self, connection, faiss_ids):
        try:
            cursor = self.get_cursor(connection)
            formatted_ids = ", ".join([str(v) for v in faiss_ids])
            query = """with a as (select o.*, f.faiss_id from faiss f left join ordinals o on f.sha256=o.sha256 where f.faiss_id in ({li}))
                       select * from a where sequence_number in (select min(sequence_number) from a group by sha256) order by FIELD(faiss_id, {li})"""
            cursor.execute(query.format(li=formatted_ids))
            rows = cursor.fetchall()
            return rows
        except Error as e:
            print("Error while retrieving numbers by faiss_id", e)


    # 2. Index functions
    def get_index(self, d):
        try:
            index = faiss.read_index("index.bin")
        except RuntimeError as e:
            print("Couldn't find index, creating new one.", e)
            index = faiss.IndexFlatIP(d)

        return index


    def add_embeddings(self, index, model, db_conn, rows):
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
                self.insert_faiss_mapping(db_conn, id_list)
            except TypeError as e:
                print(str(e) + " - trying one at a time")
                self.add_embeddings_single(index, model, db_conn, rows)
            except ValueError as e:
                print(str(e) + " - trying one at a time")
                self.add_embeddings_single(index, model, db_conn, rows)
            except OSError as e:
                print(str(e) + " - trying one at a time")
                self.add_embeddings_single(index, model, db_conn, rows)
            except:
                e = sys.exc_info()[0]
                print("Unknown Error: " + str(e) + " - trying one at a time")
                self.add_embeddings_single(index, model, db_conn, rows)


    def add_embeddings_single(self, index, model, db_conn, rows):
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
                    self.insert_faiss_mapping(db_conn, [(sha256, faiss_id)])
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


    def update_index(self, index, conn, model):
        print("Indexer starting in background..")
        last_content_id = self.get_last_insert_content_id(conn, index.ntotal)
        while True:
            if self.stop_indexer:
                print("Exiting indexer")
                break
            start_content_id = last_content_id + 1
            print(start_content_id)
            rows = self.get_image_list(conn, start_content_id)
            if len(rows) == 0:
                print("index up to date, sleeping for 60s")
                time.sleep(60)
                continue
            self.add_embeddings(index, model, conn, rows)
            self.write_index(index)
            last_content_id = rows[-1][0]
        print("Indexer exited")


    def get_text_to_image_shas(self, index, conn, search_term, n=5):
        query_emb = self.model.encode([search_term])
        D, I = index.search(query_emb, n)
        rows = self.get_shas_by_faiss_id(conn, I[0])
        zipped = list(map(lambda x, y: (x[0], x[1], float(y)), rows, D[0]))
        return zipped

    def get_image_to_image_shas(self, index, conn, image_binary, n=5):
        image_stream = BytesIO(image_binary)
        image_emb = self.model.encode([Image.open(image_stream)])
        D, I = index.search(image_emb, n)
        rows = self.get_shas_by_faiss_id(conn, I[0])
        zipped = list(map(lambda x, y: (x[0], x[1], float(y)), rows, D[0]))
        return zipped

    def get_text_to_inscription_numbers(self, model, index, conn, search_term, n=5):
        t0 = time.time()
        query_emb = model.encode([search_term])
        D, I = index.search(query_emb, n)
        t1 = time.time()
        rows = self.get_numbers_by_faiss_id(conn, I[0])
        zipped = list(map(lambda x, y: (x + (float(y),)), rows, D[0]))
        t2 = time.time()
        print("db: " + str(t2-t1) + ". index: " + str(t1-t0))
        return zipped

    def get_image_to_inscription_numbers(self, model, index, conn, image_binary, n=5):
        t0 = time.time()
        image_stream = BytesIO(image_binary)
        image_emb = model.encode([Image.open(image_stream)])
        D, I = index.search(image_emb, n)
        t1 = time.time()
        rows = self.get_numbers_by_faiss_id(conn, I[0])
        zipped = list(map(lambda x, y: (x + (float(y),)), rows, D[0]))
        t2 = time.time()
        print("db: " + str(t2-t1) + ". index: " + str(t1-t0))
        return zipped


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
config_path = "ord.yaml"
# config_path = os.getenv("DISCOVER_CONFIG_PATH")
discover = Discover()
model = SentenceTransformer('clip-ViT-B-32')
index = discover.get_index(512)
conn = discover.get_db_connection(config_path)
discover.create_faiss_mapping_table(conn)
app = Flask(__name__)

@app.route("/")
async def hello_world():
    return "Hello, World!"


@app.route("/search/<search_term>")
def search(search_term):
    rows = discover.get_text_to_inscription_numbers(model, index, conn, search_term, 5)
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
    rows = discover.get_image_to_inscription_numbers(model, index, conn, image_binary, 5)
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
    index_thread = threading.Thread(target=discover.update_index, args=(index, conn, model))
    index_thread.start()
    logger = logging.getLogger('waitress')
    logger.setLevel(logging.INFO)
    waitress.serve(app, host="0.0.0.0", port=4080)
    discover.stop_indexer = True
    print("Flask exited, waiting on index thread to finish..")
    index_thread.join()