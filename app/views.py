
"""
Entry point for web app
"""
from flask import (request, render_template, flash,
                   send_from_directory, abort)
from . import APP, rnn
import sys
sys.path.append("..")
import models2

DEFAULT_PLOT = "Once upon a time... there was a lad"


@APP.route('/static/js/<path:path>')
def send_js(path):
    """Serve our js files"""
    return send_from_directory('static/js', path)

@APP.route('/static/css/<path:path>')
def send_css(path):
    """Serve our css files"""
    return send_from_directory('static/css', path)

@APP.route('/static/img/<path:path>')
def send_img(path):
    """Serve our img files"""
    return send_from_directory('static/img', path)


@APP.route('/', methods=['GET', 'POST'])
def index():
    """
    Gets index page for viewing or handles posting a movie description and returning
    the image generated for it

    Parameters:
        description (string) (POST): Post parameter for movie description 
    """
    if request.method == 'GET':
        return default_page()
    
    level = request.form['range']
    level = float(level) / 100.0
    print(level)
    plot = rnn.get_a_plot(level)
    return render_template('index.html', plot=plot)


def default_page():
    return render_template('index.html', plot=DEFAULT_PLOT)
