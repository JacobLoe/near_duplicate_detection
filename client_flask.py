import os
from flask import Flask, flash, request, redirect, url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename

import requests
from PIL import Image
from io import BytesIO
import base64

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


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    upload_image = '''<!doctype html><title>Upload new File</title><h1>Upload new File</h1><form method=post enctype=multipart/form-data><input type=file name=file><input type=submit value=Upload></form>'''
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

            target_image = Image.open(target_image)
            buf = BytesIO()
            target_image.save(buf, 'PNG')
            target_image = buf.getvalue()
            target_image = base64.encodebytes(target_image).decode('ascii')

            html_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.html')

            url = 'http://localhost:8000/'

            server_options = {}
            try:
                r = requests.get(url)
                if r.status_code == 200:
                    data = r.json()
                    caps = data.get('data', {}).get('capabilities', {})
                    for n in ('minimum_batch_size', 'maximum_batch_size', 'available_models'):
                        server_options[n] = caps.get(n, None)
            except requests.exceptions.RequestException:
                print('try failed')
                pass

            # the response is what is sent to the server
            # Use a requests.session to use a KeepAlive connection to the server
            session = requests.session()
            headers = {"Content-Type": "application/json", "Accept": "application/json"}

            # call the main function of the query
            num_cores = 8
            num_results = 30
            with graph.as_default():

                response = session.post(url, headers=headers, json={
                    'target_image': target_image,
                    'num_results': num_results,
                    'num_cores': num_cores
                })

                output = response.json()

                # get the results from the server
                concepts = output.get('data')
            return upload_image #redirect(url_for('uploaded_file', filename=html_path))
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
