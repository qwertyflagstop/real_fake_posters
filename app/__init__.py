"""
Initialize our flask app
"""
from flask import Flask

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
APP = Flask(__name__)
APP.config['UPLOAD_FOLDER'] = 'app/uploads'
APP.secret_key = "doshmajhan"

from .views import *