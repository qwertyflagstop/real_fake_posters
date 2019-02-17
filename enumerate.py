import omdb
import threading
import time

MOVE_LIST = "movie_ids.list"
START = 75000
INDEX = START


def increment():
    global INDEX
    INDEX += 1

def check_id():
    year = 0
    while year < 2017:
        time.sleep(0.05)
        fullID = "tt{:07d}".format(INDEX)
        try:
            response = omdb.get_plot(fullID)
            print(response)
            year = int(response['Year'])
            print("Year: {} ID: {}".format(year, fullID))

            with open(MOVE_LIST, 'a+') as f:
                #with lock:
                f.write("{}\n".format(fullID))
                
        except Exception as e:
            print(e)
            pass

        increment()
        

if __name__ == '__main__':
    """
    threads = list()
    lock = threading.Lock() 
    for _ in range(10):
        threads.append(threading.Thread(target=check_id, args=(lock,)))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    """
    check_id()