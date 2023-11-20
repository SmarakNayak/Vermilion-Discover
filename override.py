import os
import sys
import csv
import mysql.connector as mysql
from mysql.connector import Error
import yaml
from sqlalchemy.engine import make_url


class Overrider:
    # 1. DB functions
    def initialize_db_pool(self, config_path):
        config = yaml.safe_load(open(config_path))
        db_config = make_url(config["db_connection_string"])
        conn = mysql.connect(host=db_config.host,
                             user=db_config.username,
                             password=db_config.password,
                             port=db_config.port,
                             database=db_config.database,
                             pool_name="overrider_pool",
                             pool_size=1)
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

config_path = os.getenv("DISCOVER_CONFIG_PATH")
if config_path is None:
    config_path = "ord.yaml"
overrider = Overrider()
conn = overrider.initialize_db_pool(config_path)
overrider.create_content_moderation_table()

with open('override.csv') as f:
    next(f)
    data = [tuple(line) for line in csv.reader(f)]
overrider.insert_moderation_overrides(data)

with open('override.csv') as f:
    next(f)
    blocked_shas = []
    for line in csv.reader(f):
        sha = line[0]
        flag = line[1]
        if "BLOCKED" in flag:
            blocked_shas.append((sha,))
    overrider.delete_blocked_content(blocked_shas)