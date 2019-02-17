
"""
Entry point for web app
"""
import os
import base64
from flask import (request, render_template, flash,
                   send_from_directory, abort)
from werkzeug.utils import secure_filename
from . import APP, ALLOWED_EXTENSIONS

DEFAULT_TITLE = "Your Title Here"
DEFAULT_PLOT = "Once upon a time... there was a lad"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    
    if 'file' not in request.files:
        flash('No file part')
        return default_page()

    file = request.files['file']

    if not allowed_file(file.filename):
        flash('Extension not allowed')
        return default_page()

    #filename = secure_filename(file.filename)
    #file.save(os.path.join(APP.config['UPLOAD_FOLDER'], filename))
    file.seek(0)
    file_raw = file.read()
    title, plot = generate_plot_and_title(file_raw)
    base64_file = base64.b64encode(file_raw)
    return render_template('index.html', title=title, plot=plot, img_data=base64_file.decode())


def default_page():
    return render_template('index.html', title=DEFAULT_TITLE, plot=DEFAULT_PLOT)


def generate_plot_and_title(file_raw):

    # do network stuff

    # replace these values with the generated ones
    return DEFAULT_TITLE, DEFAULT_PLOT