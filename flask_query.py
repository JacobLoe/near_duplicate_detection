# export FLASK_APP=flask_query.py
# flask run

import os
from flask import Flask, flash, request, redirect, url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename
import cv2

from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances

from query import get_features, write_to_html

#import threading
#threading.current_thread().name == 'MainThread'

import tensorflow as tf
graph = tf.get_default_graph()
##########################################################################

UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


class PrefixMiddleware(object):

    def __init__(self, app, prefix=''):
        self.app = app
        self.prefix = prefix

    def __call__(self, environ, start_response):

        if environ['PATH_INFO'].startswith(self.prefix):
            environ['PATH_INFO'] = environ['PATH_INFO'][len(self.prefix):]
            environ['SCRIPT_NAME'] = self.prefix
            return self.app(environ, start_response)
        else:
            start_response('404', [('Content-Type', 'text/plain')])
            return ["This url does not belong to the app.".encode()]


app = Flask(__name__)
app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix='/imagesearch')

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
        # submits an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            target_image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(target_image)
            # resize the target image for better visualization in html
            resized_file = cv2.imread(target_image)
            resized_file = cv2.resize(resized_file, (299, 299))
            cv2.imwrite(target_image, resized_file)

            # define the path to the features and the html in a way usable byy flask
            features_path = os.path.join(app.config['UPLOAD_FOLDER'], 'static/')
            html_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.html')
            
            # call the main function of the query
            num_cores = 8
            num_results = 30
            with graph.as_default():
                features, info = get_features(features_path, target_image)

            # calculate the distance for all features
            distances = pairwise_distances(features['feature_list'], features['target_feature'], metric=euclidean,n_jobs=num_cores)
            # sort by distance, ascending
            lowest_distances = sorted(
                zip(distances, info['source_video'], info['shot_begin_frame'], info['frame_timestamp'],
                    info['frame_path']))

            write_to_html(lowest_distances, html_path, num_results, target_image)

            return redirect(url_for('uploaded_file', filename=html_path))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=80)
