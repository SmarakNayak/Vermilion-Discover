import sys

import mysql.connector as mysql
from mysql.connector import Error
import yaml
from sqlalchemy.engine import make_url
import numpy as np
import faiss
from sentence_transformers import util
from PIL import Image, UnidentifiedImageError
import torch
from io import BytesIO
import time

class ImageEmbeddingContainer:
    image_embeddings = None
    content_id_sha_list = None

    def __init__(self, image_embeddings, content_id_sha_list):
        self.image_embeddings = image_embeddings
        self.content_id_sha_list = content_id_sha_list


class Discover:
    stop_indexer = False
    index = None

    # 1. DB functions
    def initialize_db_pool(self, config_path):
        config = yaml.safe_load(open(config_path))
        db_config = make_url(config["db_connection_string"])
        print(db_config)
        conn = mysql.connect(host=db_config.host,
                             user=db_config.username,
                             password=db_config.password,
                             port=db_config.port,
                             database=db_config.database,
                             pool_name="discover_pool",
                             pool_size=10)
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

    def create_content_moderation_table(self):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            cursor.execute("""CREATE TABLE IF NOT EXISTS content_moderation (            
            content_id int unsigned,
            sha256 varchar(80) not null primary key,
            automated_moderation_flag varchar(40),
            flagged_concept varchar(40),
            cosine_distance double,
            human_override_moderation_flag varchar(40),
            human_override_reason varchar(80),
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

    def insert_moderation_flag(self, moderation_details):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            query = """INSERT INTO content_moderation (content_id, sha256, automated_moderation_flag, flagged_concept, cosine_distance) VALUES (%s, %s, %s, %s, %s) 
            ON DUPLICATE KEY UPDATE content_id=VALUES(content_id), automated_moderation_flag=VALUES(automated_moderation_flag), flagged_concept=VALUES(flagged_concept), cosine_distance=VALUES(cosine_distance)"""
            cursor.executemany(query, moderation_details)
            connection.commit()
            cursor.close()
            connection.close()
        except Error as e:
            print("Error inserting moderation flags", e)
        except:
            print("unknown error")

    def insert_moderation_overrides(self, moderation_details):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            query = """INSERT INTO content_moderation (sha256, human_override_moderation_flag, human_override_reason) VALUES (%s, %s, %s) 
            ON DUPLICATE KEY UPDATE human_override_moderation_flag=VALUES(human_override_moderation_flag), human_override_reason=VALUES(human_override_reason)"""
            cursor.executemany(query, moderation_details)
            connection.commit()
            cursor.close()
            connection.close()
        except Error as e:
            print("Error inserting moderation overrides", e)
        except:
            print("unknown error")

    def delete_blocked_content(self, blocked_shas):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            query = 'UPDATE content SET content = null where sha256=%s'
            cursor.executemany(query, blocked_shas)
            connection.commit()
            cursor.close()
            connection.close()
        except Error as e:
            print("Error deleting blocked content", e)
        except:
            print("unknown error")

    def get_last_valid_faiss_id(self):
        connection = self.get_connection()
        cursor = self.get_cursor(connection)
        cursor.execute('select min(previous) from (select faiss_id, Lag(faiss_id,1) over (order BY faiss_id) as previous from faiss) a where faiss_id != previous+1')
        row = cursor.fetchone()
        cursor.close()
        connection.close()
        last_valid_id_in_db = row[0]
        return last_valid_id_in_db

    def get_faiss_db_length(self):
        connection = self.get_connection()
        cursor = self.get_cursor(connection)
        cursor.execute('select count(*) from faiss')
        row = cursor.fetchone()
        cursor.close()
        connection.close()
        if row is None or row[0] is None:
            db_length = 0
        else:
            db_length = row[0]
        return db_length

    def get_last_faiss_id_in_db(self):
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
        return last_faiss_id_in_db

    def delete_from_faiss_db(self, last_valid_id):
        connection = self.get_connection()
        cursor = self.get_cursor(connection)
        cursor.execute('Delete from faiss where faiss_id>%s', (last_valid_id,))
        connection.commit()
        cursor.close()
        connection.close()
        print("Deleted extra faiss ids from db past: " + str(last_valid_id))

    def reconcile_index_with_db(self):
        print("Reconciling index & db..")
        try:
            ##1. check max
            last_faiss_id_in_db = self.get_last_faiss_id_in_db()
            last_faiss_id_in_idx = self.index.ntotal-1
            print("last_faiss_id_in_db: " + str(last_faiss_id_in_db) + " last_faiss_id_in_idx: " + str(last_faiss_id_in_idx))

            if last_faiss_id_in_db > last_faiss_id_in_idx:
                self.delete_from_faiss_db(last_faiss_id_in_idx)
            elif last_faiss_id_in_db < last_faiss_id_in_idx:
                self.index.remove_ids(np.array([i for i in range(last_faiss_id_in_db+1, self.index.ntotal)]))
                print("Deleted extra entries from faiss")
            else:
                print("Max ids are the same.. now checking length is the same")

            ##2. check length
            db_length = self.get_faiss_db_length()
            print("DB length: " + str(db_length) + " Faiss length: " + str(self.index.ntotal))
            if db_length < self.index.ntotal:
                last_valid_id_in_db = self.get_last_valid_faiss_id()
                print("DB has gaps in index, deleting ids past first gap: " + str(last_valid_id_in_db))
                self.delete_from_faiss_db(last_valid_id_in_db)
                self.index.remove_ids(np.array([i for i in range(last_valid_id_in_db + 1, self.index.ntotal)]))
            elif db_length > self.index.ntotal:
                print("DB length longer than index length - doesn't make sense, need full reindexing")
            else:
                print("Length is same, db matches index")

            db_length = self.get_faiss_db_length()
            last_faiss_id_in_db = self.get_last_faiss_id_in_db()
            print("DB length: " + str(db_length) + " Faiss length: " + str(self.index.ntotal))
            print("DB last id: " + str(last_faiss_id_in_db) + " Faiss last id: " + str(self.index.ntotal-1))
            return self.index.ntotal-1
        except Error as e:
            print("Error while reconciling faiss_ids", e)

    def get_last_insert_content_id(self):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            cursor.execute('select max(content_id) from content where sha256 in (select sha256 from faiss where faiss_id in (select max(faiss_id) from faiss))')
            row = cursor.fetchone()
            cursor.close()
            connection.close()
            if row is None or row[0] is None:
                return -1
            else:
                return row[0]
        except Error as e:
            print("Error while retrieving last_insert", e)

    def get_image_list(self, start_number):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            cursor.execute('select content_id, sha256, content, content_type from content where content_id >= %s order by content_id limit 0, 1000',(start_number,))
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
            query = """with a as (select o.*, f.faiss_id from faiss f left join ordinals o on f.sha256=o.sha256 where f.faiss_id in ({li})),
                       b as (select min(sequence_number) as sequence_number from a group by sha256)
                       select a.* from a, b where a.sequence_number in (b.sequence_number) order by FIELD(faiss_id, {li})"""
            # query = """with a as (select o.*, f.faiss_id from faiss f left join ordinals o on f.sha256=o.sha256 where f.faiss_id in ({li})),
            #            select * from a where sequence_number in (select min(sequence_number) from a group by sha256) order by FIELD(faiss_id, {li})"""
            cursor.execute(query.format(li=formatted_ids))
            rows = cursor.fetchall()
            cursor.close()
            connection.close()
            return rows
        except Error as e:
            print("Error while retrieving numbers by faiss_id", e)

    def get_numbers_by_dbclass(self, dbclass, n):
        try:
            connection = self.get_connection()
            cursor = self.get_cursor(connection)
            query = """with a as (select sha256 from dbscan where dbscan_class = %s limit %s), 
                       b as (select min(sequence_number) as sequence_number from ordinals o, a where o.sha256 in (a.sha256) group by o.sha256)
                       select o.* from ordinals o, b where o.sequence_number in (b.sequence_number)"""
            cursor.execute(query, (dbclass, n))
            rows = cursor.fetchall()
            cursor.close()
            connection.close()
            return rows
        except Error as e:
            print("Error while retrieving numbers by dbclass", e)

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
            self.index = faiss.read_index("ivf_index.bin")
        except RuntimeError as e:
            if "No such file or directory" in str(e):
                ## initialize as flat index. On next retrain it will be converted into an IVF
                print("Couldn't find index, creating new one.", e)
                self.index = faiss.IndexFlatIP(d)
            else:
                raise
        return self.index

    def nsfw_filter(self, model, embedding_container):
        # List of standard NSFW concepts to filter out
        concepts = ['sexual', 'nude', 'sex', '18+', 'naked', 'nsfw', 'porn', 'explicit content', 'uncensored', 'gore']
        # List of special concepts, focusing on protecting images of minors
        special_concepts = ["little girl", "little boy", "young child", "young girl", "young boy", "child", "teen"]
        combined_concepts = concepts + special_concepts
        concept_embeddings = model.encode(combined_concepts)
        cos_scores = util.cos_sim(embedding_container.image_embeddings, concept_embeddings)

        THRESHOLD = 0.27
        # check if any concepts are above threshold
        max_score = torch.amax(cos_scores, 1)
        clean_index = (max_score < THRESHOLD).tolist()
        nsfw_index = (max_score >= THRESHOLD).tolist()
        max_score_index = torch.argmax(cos_scores, 1)
        max_concepts = [combined_concepts[i] for i in max_score_index]

        # update content moderation db
        moderation_list = []
        for i in range(len(embedding_container.content_id_sha_list)):
            if max_score[i] >= THRESHOLD:
                if max_score_index[i] <= 9:
                    moderation_list.append((embedding_container.content_id_sha_list[i][0],
                                            embedding_container.content_id_sha_list[i][1],
                                            "BLOCKED_NSFW_AUTOMATED",
                                            max_concepts[i],
                                            float(max_score[i])))
                else:
                    moderation_list.append((embedding_container.content_id_sha_list[i][0],
                                            embedding_container.content_id_sha_list[i][1],
                                            "BLOCKED_MINOR_AUTOMATED",
                                            max_concepts[i],
                                            float(max_score[i])))
            else:
                moderation_list.append((embedding_container.content_id_sha_list[i][0],
                                        embedding_container.content_id_sha_list[i][1],
                                        "SAFE_AUTOMATED",
                                        max_concepts[i],
                                        float(max_score[i])))

        self.insert_moderation_flag(moderation_list)

        # only pass clean images
        filtered_embeddings = embedding_container.image_embeddings[clean_index]
        filtered_content_id_sha_list = [i for (i,v) in zip(embedding_container.content_id_sha_list, clean_index) if v]
        filtered_container = ImageEmbeddingContainer(filtered_embeddings, filtered_content_id_sha_list)
        return filtered_container

    def add_embeddings(self, model, rows):
        image_list = []
        id_list = []
        content_id_list = []
        unscanned_moderation_details = []
        next_faiss_id = self.index.ntotal
        for row in rows:
            content_id = row[0]
            sha256 = row[1]
            binary_content = row[2]
            content_type = row[3]
            if "image/" in content_type and "svg" not in content_type:
                faiss_id = next_faiss_id
                image_stream = BytesIO(binary_content)
                try:
                    image_list.append(Image.open(image_stream))
                    content_id_list.append((content_id, sha256))
                except UnidentifiedImageError as e:
                    print("Couldn't open sha256: " + sha256 + ". Not a valid image")
                    unscanned_moderation_details.append((content_id, sha256, "UNKNOWN_AUTOMATED", "", 0))
                    continue
                except:
                    e = sys.exc_info()[0]
                    print(e)
                    unscanned_moderation_details.append((content_id, sha256, "UNKNOWN_AUTOMATED", "", 0))
                    continue
            else:
                unscanned_moderation_details.append((content_id, sha256, "SAFE_AUTOMATED", "", 0))

        if len(image_list) > 0:
            try:
                img_emb = model.encode(image_list)
                embeddings_container = ImageEmbeddingContainer(img_emb, content_id_list)
                filtered_container = self.nsfw_filter(model, embeddings_container)
                if len(filtered_container.image_embeddings) > 0:
                    self.index.add(filtered_container.image_embeddings)
                    for content_id_sha in filtered_container.content_id_sha_list:
                        id_list.append((content_id_sha[1], next_faiss_id))
                        next_faiss_id += 1
                    self.insert_faiss_mapping(id_list)
            except TypeError as e:
                print(str(e) + " - trying one at a time")
                self.add_embeddings_single(model, rows)
            except ValueError as e:
                print(str(e) + " - trying one at a time")
                self.add_embeddings_single(model, rows)
            except OSError as e:
                print(str(e) + " - trying one at a time")
                self.add_embeddings_single(model, rows)
            except Image.DecompressionBombError as e:
                print(str(e) + " - trying one at a time")
                self.add_embeddings_single(model, rows)
            except KeyError as e:
                print(str(e) + " - trying one at a time")
                self.add_embeddings_single(model, rows)
            except SyntaxError as e:
                print(str(e) + " - trying one at a time")
                self.add_embeddings_single(model, rows)
            except:
                e = sys.exc_info()[0]
                print("Unknown Error: " + str(e) + " - trying one at a time")
                self.add_embeddings_single(model, rows)
        self.insert_moderation_flag(unscanned_moderation_details)

    def add_embeddings_single(self, model, rows):
        next_faiss_id = self.index.ntotal
        for row in rows:
            content_id = row[0]
            sha256 = row[1]
            binary_content = row[2]
            content_type = row[3]
            if "image/" in content_type and "svg" not in content_type:
                faiss_id = next_faiss_id
                image_stream = BytesIO(binary_content)
                try:
                    img_emb = model.encode([Image.open(image_stream)])
                    embeddings_container = ImageEmbeddingContainer(img_emb, [(content_id, sha256)])
                    filtered_container = self.nsfw_filter(model, embeddings_container)
                    if len(filtered_container.image_embeddings) > 0:
                        self.index.add(filtered_container.image_embeddings)
                        self.insert_faiss_mapping([(sha256, faiss_id)])
                        next_faiss_id += 1
                except UnidentifiedImageError as e:
                    print("Couldn't open sha256: " + sha256 + ". Not a valid image")
                    self.insert_moderation_flag([(content_id, sha256, "UNKNOWN_AUTOMATED", "", 0)])
                    continue
                except TypeError as e:
                    print("TypeError: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                    self.insert_moderation_flag([(content_id, sha256, "UNKNOWN_AUTOMATED", "", 0)])
                    continue
                except ValueError as e:
                    print("ValueError: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                    self.insert_moderation_flag([(content_id, sha256, "UNKNOWN_AUTOMATED", "", 0)])
                    continue
                except OSError as e:
                    print("OSError: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                    self.insert_moderation_flag([(content_id, sha256, "UNKNOWN_AUTOMATED", "", 0)])
                    continue
                except Image.DecompressionBombError as e:
                    print("DecompressionBombError: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                    self.insert_moderation_flag([(content_id, sha256, "UNKNOWN_AUTOMATED", "", 0)])
                    continue
                except KeyError as e:
                    print("KeyError: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                    self.insert_moderation_flag([(content_id, sha256, "UNKNOWN_AUTOMATED", "", 0)])
                    continue
                except SyntaxError as e:
                    print("SyntaxError: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                    self.insert_moderation_flag([(content_id, sha256, "UNKNOWN_AUTOMATED", "", 0)])
                    continue
                except:
                    e = sys.exc_info()[0]
                    print("Unknown Error: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                    self.insert_moderation_flag([(content_id, sha256, "UNKNOWN_AUTOMATED", "", 0)])
                    raise
                    continue
            else:
                self.insert_moderation_flag([(content_id, sha256, "SAFE_AUTOMATED", "", 0)])

    def write_index(self):
        faiss.write_index(self.index, "ivf_index.bin")

    def update_index(self, model):
        print("Indexer starting in background..")
        self.reconcile_index_with_db()
        last_content_id = self.get_last_insert_content_id()
        last_retrain_id = last_content_id
        while True:
            time.sleep(1) #Helps with debugging for some reason
            t0 = time.perf_counter()
            if self.stop_indexer:
                print("Exiting indexer, writing index")
                self.write_index()
                print("Index written to drive")
                break
            start_content_id = last_content_id + 1
            print(start_content_id)
            rows = self.get_image_list(start_content_id)
            t1 = time.perf_counter()
            if rows is None:
                print("Trying again in 60s")
                time.sleep(60)
                continue
            if len(rows) == 0:
                print("index up to date, sleeping for 60s")
                time.sleep(60)
                continue
            self.add_embeddings(model, rows)
            t2 = time.perf_counter()
            print("add: " + str(t2 - t1) + ". get images: " + str(t1 - t0))
            last_content_id = rows[-1][0]
            if last_content_id > last_retrain_id + 100000:
                t1 = time.perf_counter()
                self.retrain_index()
                t2 = time.perf_counter()
                self.write_index()
                t3 = time.perf_counter()
                last_retrain_id = last_content_id
                print("write index: " + str(t3 - t2) + ". " + ". retrain index: " + str(t2 - t1))
        print("Indexer exited")

    def retrain_index(self):
        print("Retraining index")
        length = self.index.ntotal
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.make_direct_map()
        embeddings = self.index.reconstruct_n(0, length)
        nlist = 128
        d = 512
        quantizer = faiss.IndexFlatIP(d)
        new_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        print("Training")
        new_index.train(embeddings)
        print("Adding")
        new_index.add(embeddings)
        new_index.nprobe=10
        print("Finished")
        self.index = new_index

    def get_text_to_image_shas(self, model, search_term, n=5):
        query_emb = model.encode([search_term])
        D, I = self.index.search(query_emb, n)
        rows = self.get_shas_by_faiss_id(I[0])
        zipped = list(map(lambda x, y: (x[0], x[1], float(y)), rows, D[0]))
        return zipped

    def get_image_to_image_shas(self, model, image_binary, n=5):
        image_stream = BytesIO(image_binary)
        image_emb = model.encode([Image.open(image_stream)])
        D, I = self.index.search(image_emb, n)
        rows = self.get_shas_by_faiss_id(I[0])
        zipped = list(map(lambda x, y: (x[0], x[1], float(y)), rows, D[0]))
        return zipped

    def get_text_to_inscription_numbers(self, model, search_term, n=5):
        t0 = time.time()
        query_emb = model.encode([search_term])
        t1 = time.time()
        D, I = self.index.search(query_emb, n)
        t2 = time.time()
        rows = self.get_numbers_by_faiss_id(I[0])
        zipped = list(map(lambda x, y: (x + (float(y),)), rows, D[0]))
        t3 = time.time()
        print("db: " + str(t3-t2) + ". index: " + str(t2-t1) + ". encode: " + str(t1-t0))
        return zipped

    def get_image_to_inscription_numbers(self, model, image_binary, n=5):
        t0 = time.time()
        image_stream = BytesIO(image_binary)
        image_emb = model.encode([Image.open(image_stream)])
        t1 = time.time()
        D, I = self.index.search(image_emb, n)
        t2 = time.time()
        rows = self.get_numbers_by_faiss_id(I[0])
        zipped = list(map(lambda x, y: (x + (float(y),)), rows, D[0]))
        t3 = time.time()
        print("db: " + str(t3-t2) + ". index: " + str(t2-t1) + ". encode: " + str(t1-t0))
        return zipped

    def get_dbclass_to_inscription_numbers(self, dbclass, n):
        t0 = time.time()
        rows = self.get_numbers_by_dbclass(dbclass, n)
        t1 = time.time()
        print("db: " + str(t1 - t0))
        return rows

    def get_similar_images(self, model, sha256, n=5):
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
                D, I = self.index.search(image_emb, n)
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

