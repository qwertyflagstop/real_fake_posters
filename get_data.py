from h5py import File, special_dtype
import json
from Queue import Queue
from threading import Thread, Lock
from pprint import pprint
from io import BytesIO
import time
import numpy as np
from omdb import get_plot, get_poster
import random

def fetch_poster(queue, lock, w_id):
    global ids
    global worker_count
    global index
    global tt_index
    while True:
        lock.acquire()
        # if index >= len(ids):
        #     worker_count -= 1
        #     lock.release()
        #     return
        i = tt_index
        tt_index += 1
        lock.release()
        id = 'tt{:07d}'.format(i)
        # if (i+w_id)%100==0:
        #     print('sleeping')
        #     time.sleep(np.random.randint(5,6))
        time.sleep(random.uniform(0,0.5))
        try:
            plot = get_plot(id)
            if plot:
                poster = get_poster(id, height=640)
                if poster:
                    queue.put({'poster': poster, 'plot': plot, 'id': id})
            else:
                print('NO DICE')
        except KeyError:
            print('no plot for {}'.format(id))
            continue
        except Exception as e:
            print e
            continue

class MovieDB(object):

    def __init__(self, name):
        super(MovieDB, self).__init__()
        self.name = name
        path = '/media/qwertyflagstop/data/{}.h5'.format(self.name)
        self.file = File(path, mode='a')

    def download_songs_from_list_in_file(self, file_name, num_workers = 8):
        """

            :param file_name: the name of file with IMDB ids
            :return: nothing, it downloads stuff
        """
        global ids
        global worker_count
        global index
        global tt_index
        with open(file_name, 'r') as fp:
            ids = json.load(fp)
        tt_index = 75000
        for k in self.file.keys(): #remove any we already got
            tt_index = max(tt_index,int(k[2:]))
        worker_count = num_workers
        index = 0
        queue = Queue()
        lock = Lock()
        for j in np.arange(0, num_workers):
            t = Thread(target=fetch_poster, args=[queue, lock, j])
            t.daemon = True
            t.start()
        while worker_count>0:
            try:
                print('got {} movies'.format(len(self.file.keys())))
                s = queue.get(timeout=4)
                poster_bytes = np.array(s['poster'])
                plot = np.string_(s['plot'])
                self.file.create_dataset('{}/poster'.format(s['id']), data=poster_bytes)
                self.file.create_dataset('{}/plot'.format(s['id']), data=plot)
                self.file.flush()
                exit()
            except:
                continue
        print('DONE!')
        self.file.flush()
        self.file.close()

if __name__ == '__main__':
    db = MovieDB('movies')
    db.download_songs_from_list_in_file('movie_ids.json')