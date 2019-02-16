"""
All the data
"""
import requests

KEY_FILE = "api.key"
KEY = open(KEY_FILE).read()
DATA_API = "http://www.omdbapi.com/?apikey={}&".format(KEY)
POSTER_API = "http://img.omdbapi.com/?apikey={}&".format(KEY)


def get_plot(imdbID):
    """
    Gets the plot of the given movie

    Parameters:
        imdbID (string): imdb ID of the movie

    Returns:
        plot (string): full plot for the movie
    """
    response = requests.get("{}i={}&plot=full".format(DATA_API, imdbID))
    plot = response.json()['plot']
    return plot

def get_poster(imdbID):
    """
    Gets the poster of the given movie

    Parameters:
        imdbID (string): imdb ID of the movie

    Returns:
        poster (image): image file of the poster
    """
    response = requests.get("{}i={}".format(DATA_API, imdbID))


if __name__ == '__main__':
    test_movies = ["Hitch", "Up", "Harry Potter"]
    for movie in test_movies:
        get_plot(movie)
        get_posters(movie)