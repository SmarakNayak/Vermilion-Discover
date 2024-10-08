import asyncpg
import sys
import yaml
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
    api_pool = None
    update_pool = None
    nprobe = 32

    # 0. Initialize
    async def setup_db(self, config_path):
        await self.initialize_api_pool(config_path)
        await self.create_faiss_mapping_table()
        await self.create_content_moderation_table()

    # 1. DB functions
    async def initialize_api_pool(self, config_path):
        config = yaml.safe_load(open(config_path))
        self.api_pool = await asyncpg.create_pool(host=config["db_host"], user=config["db_user"], database=config["db_name"], password=config["db_password"])

    async def initialize_update_pool(self, config_path):
        config = yaml.safe_load(open(config_path))
        self.update_pool = await asyncpg.create_pool(host=config["db_host"], user=config["db_user"], database=config["db_name"], password=config["db_password"])

    async def create_faiss_mapping_table(self):
        try:
            async with self.api_pool.acquire() as conn:
                await conn.execute("""CREATE TABLE IF NOT EXISTS faiss (
                sha256 varchar(80) not null primary key,
                faiss_id bigint not null
                )""")
                await conn.execute("CREATE INDEX IF NOT EXISTS index_faiss_id ON faiss (faiss_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS index_faiss_sha256 ON faiss (sha256)")
        except Exception as e:
            print(f"Error creating Postgres table: {e}")

    async def create_content_moderation_table(self):
        try:
            async with self.api_pool.acquire() as conn:
                await conn.execute("""CREATE TABLE IF NOT EXISTS content_moderation (            
                content_id bigint,
                sha256 varchar(80) not null primary key,
                automated_moderation_flag varchar(40),
                flagged_concept varchar(40),
                cosine_distance double precision,
                human_override_moderation_flag varchar(40),
                human_override_reason varchar(80)
                )""")
                await conn.execute("CREATE INDEX IF NOT EXISTS index_content_moderation_sha256 ON content_moderation (sha256)")
        except Exception as e:
            print(f"Error creating Postgres table: {e}")

    async def create_dbscan_table(self):
        try:
            async with self.api_pool.acquire() as conn:
                await conn.execute("""CREATE TABLE IF NOT EXISTS dbscan (
                sha256 varchar(80) not null primary key,
                faiss_id bigint not null,
                dbscan_class bigint not null
                )""")
                await conn.execute("CREATE INDEX IF NOT EXISTS index_dbscan_class ON dbscan (dbscan_class)")
                await conn.execute("CREATE INDEX IF NOT EXISTS index_dbscan_sha256 ON dbscan (sha256)")
        except Exception as e:
            print("Error creating Postgres table: ", e)

    async def insert_dbscan_class(self, classes):
        try:
            async with self.api_pool.acquire() as conn:
                query = "INSERT INTO dbscan (sha256, faiss_id, dbscan_class) VALUES ($1, $2, $3) ON CONFLICT (sha256) DO UPDATE SET faiss_id=EXCLUDED.faiss_id, dbscan_class=EXCLUDED.dbscan_class"
                await conn.executemany(query, classes)
        except Exception as e:
            print("Error inserting dbscan class", e)

    async def get_shas_up_to_faiss_id(self, faiss_id):
        try:
            async with self.api_pool.acquire() as conn:
                rows = await conn.fetch('select sha256, faiss_id from faiss where faiss_id < $1 order by faiss_id', faiss_id)
            return rows
        except Exception as e:
            print("Error while retrieving shas", e)

    async def insert_faiss_mapping(self, mappings):
        try:
            async with self.update_pool.acquire() as conn:
                query = "INSERT INTO faiss (sha256, faiss_id) VALUES ($1, $2) ON CONFLICT (sha256) DO UPDATE SET faiss_id = EXCLUDED.faiss_id"
                await conn.executemany(query, mappings)
        except Exception as e:
            print(f"Unexpected error: {e}")

    async def insert_moderation_flag(self, moderation_details):
        try:
            async with self.update_pool.acquire() as conn:
                query = """INSERT INTO content_moderation (content_id, sha256, automated_moderation_flag, flagged_concept, cosine_distance) VALUES ($1, $2, $3, $4, $5) 
                ON CONFLICT (sha256) DO UPDATE SET content_id=EXCLUDED.content_id, automated_moderation_flag=EXCLUDED.automated_moderation_flag, flagged_concept=EXCLUDED.flagged_concept, cosine_distance=EXCLUDED.cosine_distance"""
                await conn.executemany(query, moderation_details)
        except Exception as e:
            print(f"Unexpected error: {e}")

    async def insert_moderation_overrides(self, moderation_details):
        try:
            async with self.update_pool.acquire() as conn:
                query = """INSERT INTO content_moderation (sha256, human_override_moderation_flag, human_override_reason) VALUES ($1, $2, $3) 
                ON CONFLICT (sha256) DO UPDATE SET human_override_moderation_flag=EXCLUDED.human_override_moderation_flag, human_override_reason=EXCLUDED.human_override_reason"""
                await conn.executemany(query, moderation_details)
        except Exception as e:
            print(f"Unexpected error: {e}")

    async def delete_blocked_content(self, blocked_shas):
        try:
            async with self.update_pool.acquire() as conn:
                query = 'UPDATE content SET content = null where sha256=$1'
                await conn.executemany(query, blocked_shas)
        except Exception as e:
            print(f"Unexpected error: {e}")

    async def get_last_valid_faiss_id(self):
        async with self.update_pool.acquire() as conn:
            row = await conn.fetchrow('select min(previous) from (select faiss_id, Lag(faiss_id,1) over (order BY faiss_id) as previous from faiss) a where faiss_id != previous+1')
        last_valid_id_in_db = row[0]
        return last_valid_id_in_db

    async def get_faiss_db_length(self):
        async with self.update_pool.acquire() as conn:
            row = await conn.fetchrow('select count(*) from faiss')
        if row is None or row[0] is None:
            db_length = 0
        else:
            db_length = row[0]
        return db_length

    async def get_last_faiss_id_in_db(self):
        async with self.update_pool.acquire() as conn:
            row = await conn.fetchrow('select max(faiss_id) from faiss')
        if row is None or row[0] is None:
            last_faiss_id_in_db = -1
        else:
            last_faiss_id_in_db = row[0]
        return last_faiss_id_in_db

    async def delete_from_faiss_db(self, last_valid_id):
        try:
            async with self.update_pool.acquire() as conn:
                await conn.execute('Delete from faiss where faiss_id>$1', last_valid_id)
                print("Deleted extra faiss ids from db past: " + str(last_valid_id))
        except Exception as e:
            print(f"Unexpected error: {e}")

    async def reconcile_index_with_db(self):
        print("Reconciling index & db..")
        try:
            ##1. check max
            last_faiss_id_in_db = await self.get_last_faiss_id_in_db()
            last_faiss_id_in_idx = self.index.ntotal-1
            print("last_faiss_id_in_db: " + str(last_faiss_id_in_db) + " last_faiss_id_in_idx: " + str(last_faiss_id_in_idx))

            if last_faiss_id_in_db > last_faiss_id_in_idx:
                await self.delete_from_faiss_db(last_faiss_id_in_idx)
            elif last_faiss_id_in_db < last_faiss_id_in_idx:
                self.index.remove_ids(np.array([i for i in range(last_faiss_id_in_db+1, self.index.ntotal)]))
                print("Deleted extra entries from faiss")
            else:
                print("Max ids are the same.. now checking length is the same")

            ##2. check length
            db_length = await self.get_faiss_db_length()
            print("DB length: " + str(db_length) + " Faiss length: " + str(self.index.ntotal))
            if db_length < self.index.ntotal:
                last_valid_id_in_db = await self.get_last_valid_faiss_id()
                print("DB has gaps in index, deleting ids past first gap: " + str(last_valid_id_in_db))
                await self.delete_from_faiss_db(last_valid_id_in_db)
                self.index.remove_ids(np.array([i for i in range(last_valid_id_in_db + 1, self.index.ntotal)]))
            elif db_length > self.index.ntotal:
                print("DB length longer than index length - doesn't make sense, need full reindexing")
            else:
                print("Length is same, db matches index")

            db_length = await self.get_faiss_db_length()
            last_faiss_id_in_db = await self.get_last_faiss_id_in_db()
            print("DB length: " + str(db_length) + " Faiss length: " + str(self.index.ntotal))
            print("DB last id: " + str(last_faiss_id_in_db) + " Faiss last id: " + str(self.index.ntotal-1))
            return self.index.ntotal-1
        except Exception as e:
            print("Error while reconciling faiss_ids", e)

    async def get_last_insert_content_id(self):
        try:
            async with self.update_pool.acquire() as conn:
                row = await conn.fetchrow('select max(content_id) from content where sha256 in (select sha256 from faiss where faiss_id in (select max(faiss_id) from faiss))')
            if row is None or row[0] is None:
                return -1
            else:
                return row[0]
        except Exception as e:
            print("Error while retrieving last_insert", e)

    async def get_image_list(self, start_number):
        try:
            async with self.update_pool.acquire() as conn:
                rows = await conn.fetch('select content_id, sha256, content, content_type from content where content_id >= $1 order by content_id limit 1000', start_number)
                return rows
        except Exception as e:
            print("Error while retrieving image list", e)

    async def get_shas_by_faiss_id(self, faiss_ids):
        try:
            async with self.api_pool.acquire() as conn:
                formatted_ids = ", ".join([str(v) for v in faiss_ids])
                rows = await conn.fetch('select sha256, faiss_id from faiss where faiss_id in ({li}) order by array_position(array[{li}], faiss_id)'.format(li=formatted_ids))
                return rows
        except Exception as e:
            print("Error while retrieving last_insert", e)

    async def get_numbers_by_faiss_id(self, faiss_ids):
        try:
            async with self.api_pool.acquire() as conn:
                formatted_ids = ", ".join([str(v) for v in faiss_ids])
                query = """select o.*, f.faiss_id from faiss f 
                           left join content c on f.sha256=c.sha256
                           left join ordinals_full_v o on c.content_id=o.sequence_number 
                           where f.faiss_id in ({li})
                           order by array_position(array[{li}], faiss_id)"""
                rows = await conn.fetch(query.format(li=formatted_ids))
                return rows
        except Exception as e:
            print("Error while retrieving numbers by faiss_id", e)

    async def get_numbers_by_dbclass(self, dbclass, n):
        try:
            async with self.api_pool.acquire() as conn:
                query = """with a as (select sha256 from dbscan where dbscan_class = $1 limit $2), 
                       b as (select min(sequence_number) as sequence_number from ordinals o, a where o.sha256 in (a.sha256) group by o.sha256)
                       select o.* from ordinals o, b where o.sequence_number in (b.sequence_number)"""
                rows = await conn.fetch(query, (dbclass, n))
                return rows
        except Exception as e:
            print("Error while retrieving numbers by dbclass", e)

    async def get_content_from_sha(self, sha256):
        try:
            async with self.api_pool.acquire() as conn:
                rows = await conn.fetchrow('select sha256, content, content_type from content where sha256=\'{sha}\''.format(sha=sha256))
                return rows
        except Exception as e:
            print("Error while retrieving content", e)

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

    async def nsfw_filter(self, model, embedding_container):
        # List of standard NSFW concepts to filter out
        concepts = ['sexual', 'nude', 'sex', '18+', 'naked', 'nsfw', 'porn', 'explicit content', 'uncensored', 'gore']
        # List of special concepts, focusing on protecting images of minors
        special_concepts = ["little girl", "little boy", "young child", "young girl", "young boy", "child", "teen"]
        combined_concepts = concepts + special_concepts
        concept_embeddings = model.encode(combined_concepts)
        cos_scores = util.cos_sim(embedding_container.image_embeddings, concept_embeddings)

        THRESHOLD = 0.24
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

        await self.insert_moderation_flag(moderation_list)

        # only pass clean images
        filtered_embeddings = embedding_container.image_embeddings[clean_index]
        filtered_content_id_sha_list = [i for (i,v) in zip(embedding_container.content_id_sha_list, clean_index) if v]
        filtered_container = ImageEmbeddingContainer(filtered_embeddings, filtered_content_id_sha_list)
        return filtered_container

    async def add_embeddings(self, model, rows):
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
                filtered_container = await self.nsfw_filter(model, embeddings_container)
                if len(filtered_container.image_embeddings) > 0:
                    self.index.add(filtered_container.image_embeddings)
                    for content_id_sha in filtered_container.content_id_sha_list:
                        id_list.append((content_id_sha[1], next_faiss_id))
                        next_faiss_id += 1
                    await self.insert_faiss_mapping(id_list)
            except TypeError as e:
                print(str(e) + " - trying one at a time")
                await self.add_embeddings_single(model, rows)
            except ValueError as e:
                print(str(e) + " - trying one at a time")
                await self.add_embeddings_single(model, rows)
            except OSError as e:
                print(str(e) + " - trying one at a time")
                await self.add_embeddings_single(model, rows)
            except Image.DecompressionBombError as e:
                print(str(e) + " - trying one at a time")
                await self.add_embeddings_single(model, rows)
            except KeyError as e:
                print(str(e) + " - trying one at a time")
                await self.add_embeddings_single(model, rows)
            except SyntaxError as e:
                print(str(e) + " - trying one at a time")
                await self.add_embeddings_single(model, rows)
            except:
                e = sys.exc_info()[0]
                print("Unknown Error: " + str(e) + " - trying one at a time")
                await self.add_embeddings_single(model, rows)
        await self.insert_moderation_flag(unscanned_moderation_details)

    async def add_embeddings_single(self, model, rows):
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
                    filtered_container = await self.nsfw_filter(model, embeddings_container)
                    if len(filtered_container.image_embeddings) > 0:
                        self.index.add(filtered_container.image_embeddings)
                        await self.insert_faiss_mapping([(sha256, faiss_id)])
                        next_faiss_id += 1
                except UnidentifiedImageError as e:
                    print("Couldn't open sha256: " + sha256 + ". Not a valid image")
                    await self.insert_moderation_flag([(content_id, sha256, "UNKNOWN_AUTOMATED", "", 0)])
                    continue
                except TypeError as e:
                    print("TypeError: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                    await self.insert_moderation_flag([(content_id, sha256, "UNKNOWN_AUTOMATED", "", 0)])
                    continue
                except ValueError as e:
                    print("ValueError: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                    await self.insert_moderation_flag([(content_id, sha256, "UNKNOWN_AUTOMATED", "", 0)])
                    continue
                except OSError as e:
                    print("OSError: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                    await self.insert_moderation_flag([(content_id, sha256, "UNKNOWN_AUTOMATED", "", 0)])
                    continue
                except Image.DecompressionBombError as e:
                    print("DecompressionBombError: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                    await self.insert_moderation_flag([(content_id, sha256, "UNKNOWN_AUTOMATED", "", 0)])
                    continue
                except KeyError as e:
                    print("KeyError: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                    await self.insert_moderation_flag([(content_id, sha256, "UNKNOWN_AUTOMATED", "", 0)])
                    continue
                except SyntaxError as e:
                    print("SyntaxError: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                    await self.insert_moderation_flag([(content_id, sha256, "UNKNOWN_AUTOMATED", "", 0)])
                    continue
                except:
                    e = sys.exc_info()[0]
                    print("Unknown Error: " + str(e) + " for sha256: " + sha256 + ". Skipping")
                    await self.insert_moderation_flag([(content_id, sha256, "UNKNOWN_AUTOMATED", "", 0)])
                    raise
                    continue
            else:
                await self.insert_moderation_flag([(content_id, sha256, "SAFE_AUTOMATED", "", 0)])

    def write_index(self):
        faiss.write_index(self.index, "ivf_index.bin")

    async def update_index(self, model, config_path):
        print("Indexer starting in background..")
        await self.initialize_update_pool(config_path)
        await self.reconcile_index_with_db()
        self.update_nprobe()
        last_content_id = await self.get_last_insert_content_id()
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
            rows = await self.get_image_list(start_content_id)
            t1 = time.perf_counter()
            if rows is None:
                print("Trying again in 60s")
                time.sleep(60)
                continue
            if len(rows) == 0:
                print("index up to date, sleeping for 60s")
                time.sleep(60)
                continue
            await self.add_embeddings(model, rows)
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
        d = 768
        quantizer = faiss.IndexFlatIP(d)
        new_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        print("Training")
        new_index.train(embeddings)
        print("Adding")
        new_index.add(embeddings)
        new_index.nprobe = self.nprobe
        print("Finished")
        self.index = new_index

    def update_nprobe(self):
        self.index.nprobe = self.nprobe

    async def get_text_to_image_shas(self, model, search_term, n=5):
        query_emb = model.encode([search_term])
        D, I = self.index.search(query_emb, n)
        rows = await self.get_shas_by_faiss_id(I[0])
        zipped = list(map(lambda x, y: (x[0], x[1], float(y)), rows, D[0]))
        return zipped

    async def get_image_to_image_shas(self, model, image_binary, n=5):
        image_stream = BytesIO(image_binary)
        image_emb = model.encode([Image.open(image_stream)])
        D, I = self.index.search(image_emb, n)
        rows = await self.get_shas_by_faiss_id(I[0])
        zipped = list(map(lambda x, y: (x[0], x[1], float(y)), rows, D[0]))
        return zipped

    async def get_text_to_inscription_numbers(self, model, search_term, n=5):
        t0 = time.time()
        query_emb = model.encode([search_term])
        t1 = time.time()
        D, I = self.index.search(query_emb, n)
        t2 = time.time()
        records = await self.get_numbers_by_faiss_id(I[0])
        zipped = list(map(lambda record, distance: {**dict(record), "distance": distance}, records, D[0].tolist()))
        t3 = time.time()
        print("db: " + str(t3-t2) + ". index: " + str(t2-t1) + ". encode: " + str(t1-t0))
        return zipped

    async def get_image_to_inscription_numbers(self, model, image_binary, n=5):
        t0 = time.time()
        image_stream = BytesIO(image_binary)
        image_emb = model.encode([Image.open(image_stream)])
        t1 = time.time()
        D, I = self.index.search(image_emb, n)
        t2 = time.time()
        records = await self.get_numbers_by_faiss_id(I[0])
        zipped = list(map(lambda record, distance: {**dict(record), "distance": distance}, records, D[0].tolist()))
        t3 = time.time()
        print("db: " + str(t3-t2) + ". index: " + str(t2-t1) + ". encode: " + str(t1-t0))
        return zipped

    async def get_dbclass_to_inscription_numbers(self, dbclass, n):
        t0 = time.time()
        records = await self.get_numbers_by_dbclass(dbclass, n)
        zipped = list(map(lambda record: dict(record), records))
        t1 = time.time()
        print("db: " + str(t1 - t0))
        return zipped

    async def get_similar_images(self, model, sha256, n=5):
        t0 = time.time()
        rows = await self.get_content_from_sha(sha256)
        binary_content = rows["content"]
        content_type = rows["content_type"]
        if "image/" in content_type and "svg" not in content_type:
            t1 = time.time()
            image_stream = BytesIO(binary_content)
            try:
                s1 = time.time()
                image = Image.open(image_stream)
                s2 = time.time()
                image_emb = model.encode([image])
                s3 = time.time()
                D, I = self.index.search(image_emb, n)
                t2 = time.time()
                records = await self.get_numbers_by_faiss_id(I[0])
                zipped = list(map(lambda record, distance: {**dict(record), "distance": distance}, records, D[0].tolist()))
                t3 = time.time()
                print("db1: " + str(t1 - t0) + ". index: " + str(t2 - t1) + ". db2: " + str(t3 - t2))
                print("open: " + str(s2 - s1) + ". encode: " + str(s3 - s2) + ". search: " + str(t2 - s3))
                return zipped
            except UnidentifiedImageError as e:
                print("Couldn't open sha256: " + sha256 + ". Not a valid image")
                return
            except:
                e = sys.exc_info()[0]
                print(e)
                return

