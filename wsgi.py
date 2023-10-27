##Entrance for gunicorn
from main import app, discover, model, index
import threading

index_thread = threading.Thread(target=discover.update_index, args=(index, model))
index_thread.start()

if __name__ == "__main__":
    print("wsgi main hit")
    app.run()
    