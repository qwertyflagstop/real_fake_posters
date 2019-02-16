"""
All the data
"""
import requests

KEY_FILE = "api.key"
KEY = open(KEY_FILE).read()
DATA_API = "http://www.omdbapi.com/?apikey={}&".format(KEY)
POSTER_API = "http://img.omdbapi.com/?apikey={}&".format(KEY)


def get_data():
    pass

def get_posters():
    pass

if __name__ == '__main__':
    get_data()
    get_posters()