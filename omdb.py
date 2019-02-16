"""
All the data
"""
import json
import requests
import shutil


KEY_FILE = "api.key"
KEY = open(KEY_FILE).read()
DATA_API = "http://www.omdbapi.com/?apikey={}&".format(KEY)
POSTER_API = "http://img.omdbapi.com/?apikey={}&".format(KEY)
ID_FILE = "movie_ids.json"


def get_plot(imdbID):
    """
    Gets the plot of the given movie

    Parameters:
        imdbID (string): imdb ID of the movie

    Returns:
        plot (string): full plot for the movie
    """
    response = requests.get("{}i={}&plot=full".format(DATA_API, imdbID))
    return response.json()['Plot'] if response.status_code == 200 else None 

def get_poster(imdbID):
    """
    Gets the poster of the given movie

    Parameters:
        imdbID (string): imdb ID of the movie

    Returns:
        poster (image): image file of the poster
    """
    response = requests.get("{}i={}".format(POSTER_API, imdbID), stream=True)
    """
    if response.status_code == 200:
        with open("{}.jpg".format(imdbID), 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
    """

    return response.raw if response.status_code == 200 else None

if __name__ == '__main__':
    ids = json.load(open(ID_FILE))
    for i in ids:
        i = "tt{}".format(i)
        plot = get_plot(i)
        poster = get_poster(i)