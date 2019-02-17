"""
Initialize our flask app
"""
import os
import sys
from flask import Flask
sys.path.append("..")
import models2


ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
APP = Flask(__name__)
APP.config['UPLOAD_FOLDER'] = 'app/uploads'
APP.secret_key = "doshmajhan"

# load model
t_file = models2.TextFile('plots.txt')
n = os.path.basename(t_file.fp).replace('.txt','')
rnn = models2.CharRNN(n,t_file.vocab_size,512, t_file,128,0.9)
print(rnn.model.summary())
rnn.load_models()

from .views import *
