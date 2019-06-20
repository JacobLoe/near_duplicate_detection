# export FLASK_APP=flask_query.py
# flask run

import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from query import main
import cv2
##########################################################################
UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(file_path)
            #resize the target image for better visualization in html
            resized_file = cv2.imread(file_path)      
            resized_file = cv2.resize(resized_file,(299,299))
            resized_file = cv2.imwrite(file_path,resized_file)

            features_path = os.path.join(app.config['UPLOAD_FOLDER'],'static/')
            html_path = os.path.join(app.config['UPLOAD_FOLDER'],'results.html')
            
            #call the main function of the query
            main(features_path,file_path,html_path)

            with open(html_path,'r') as f:
                 html_string = f.read()

            return redirect(url_for('uploaded_file',filename=html_path))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

from flask import send_from_directory

@app.route('/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0',port=80)
