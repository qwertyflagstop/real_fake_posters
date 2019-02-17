
"""
Entry point for web app
"""
import os
from flask import (request, render_template, flash,
                   send_from_directory, abort)
from werkzeug.utils import secure_filename
from . import APP, ALLOWED_EXTENSIONS

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
        return render_template('index.html')
    
    if 'file' not in request.files:
        flash('No file part')
        return render_template('index.html')

    file = request.files['file']

    if not allowed_file(file.filename):
        flash('Extension not allowed')
        return render_template('index.html')

    filename = secure_filename(file.filename)
    file.save(os.path.join(APP.config['UPLOAD_FOLDER'], filename))
    
    return render_template('index.html')
