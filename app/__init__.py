from flask import Flask


app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'app/static/uploads'
from skin_app import routes 