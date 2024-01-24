from sklearn.cluster import DBSCAN
import os
import sys
from sentence_transformers import SentenceTransformer
import faiss
import mysql.connector as mysql
from mysql.connector import Error
import yaml
from sqlalchemy.engine import make_url
import datetime


class Cluster:

    # 1. DB functions
    def initialize_db_pool(self, config_path):
        config = yaml.safe_load(open(config_path))
        db_config = make_url(config["db_connection_string"])
        conn = mysql.connect(host=db_config.host,
                             user=db_config.username,
                             password=db_config.password,
                             port=db_config.port,
                             database=db_config.database,
                             pool_name="cluster_pool",
                             pool_size=5,
                             use_pure=True,
                             ssl_disabled=True)
        return conn

    def get_connection(self):
        connection = mysql.connect(pool_name="cluster_pool")
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

    def create_dbscan_table(self):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            cursor.execute("""CREATE TABLE IF NOT EXISTS dbscan (
            sha256 varchar(80) not null primary key,
            faiss_id bigint unsigned not null,
            dbscan_class bigint not null,
            INDEX index_dbscan (dbscan_class),
            INDEX sha256 (sha256)
            )""")
            cursor.close()
            connection.close()
        except Error as e:
            print("Error creating MySQL table", e)

    def insert_dbscan_class(self, classes):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            query = "INSERT INTO dbscan (sha256, faiss_id, dbscan_class) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE faiss_id=VALUES(faiss_id), dbscan_class=VALUES(dbscan_class)"
            cursor.executemany(query, classes)
            connection.commit()
            cursor.close()
            connection.close()
        except Error as e:
            print("Error inserting dbscan class", e)
        except:
            print("unknown error")

    def get_shas_up_to_faiss_id(self, faiss_id):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            cursor.execute('select sha256, faiss_id from faiss where faiss_id < %s order by faiss_id', (faiss_id,))
            rows = cursor.fetchall()
            cursor.close()
            connection.close()
            return rows
        except Error as e:
            print("Error while retrieving last_insert", e)

    # 2. Index functions
    def get_index(self, d):
        index = faiss.read_index("ivf_index.bin")
        if isinstance(index, faiss.IndexIVFFlat):
            index.make_direct_map()
        return index


now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
config_path = os.getenv("DISCOVER_CONFIG_PATH")
if config_path is None:
    config_path = "ord.yaml"
cluster = Cluster()
model = SentenceTransformer('clip-ViT-B-32')
index = cluster.get_index(512)
conn = cluster.initialize_db_pool(config_path)
cluster.create_dbscan_table()
print("db initialized")
length = index.ntotal
print(index.ntotal)
faiss_ids = list(range(0, length))
embeddings = index.reconstruct_n(0, length)
print("embeddings reconstructed")
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

ordb = DBSCAN(eps=0.1, min_samples=10, metric="cosine")
ordb.fit(embeddings)
labels = ordb.labels_.tolist()
print("dbscan complete")
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

shas = cluster.get_shas_up_to_faiss_id(length)
print("shas retreieved")
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

combined = list(map(lambda x, y: (x[0], x[1], y), shas, labels))
cluster.insert_dbscan_class(combined)
print("classes inserted")
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))