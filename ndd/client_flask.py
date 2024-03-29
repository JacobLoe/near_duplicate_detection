import os
from flask import Flask, flash, request, redirect
from flask import send_from_directory
from werkzeug.utils import secure_filename

import requests
from PIL import Image
from io import BytesIO
import base64
import hashlib
from functools import partial
import time
import logging

from waitress import serve


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False  # prevent log messages from appearing twice

UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# default adress should be 'server_ndd'
if 'IMAGESEARCH_HOST' in os.environ:
    IMAGESEARCH_HOST = os.environ['IMAGESEARCH_HOST']
else:
    IMAGESEARCH_HOST = 'server_ndd'


def write_html_str(results, query_image):

    # write to html
    html_str = '<!DOCTYPE html><html lang="en"><table cellspacing="20"><tr><th>thumbnail</th><th>videofile</th><th>frame timestamp</th><th>shot_beginning</th><th>distance</th></tr>'
    # add the target image to the html
    html_str += str('<tr><td><img src="data:image/jpg;base64,{query_image}" width="480"></td></tr>'.format(query_image=query_image))

    # append the found images to the html
    for row in results:
        # write an entry for the table, format is: frame_path, source_video, frame_timestamp, shot_begin_frame, distance
        html_str += str('<tr><td><img src="data:image/jpg;base64,{frame}" width="480"></td><td>{source_video}</td>'
                        '<td>{frame_timestamp}</td><td>{shot_begin_timestamp}</td><td>{distance}</td></tr>'.format(frame=row['frame'],
                                                                                            source_video=row['source_video'],
                                                                                            frame_timestamp=row['frame_timestamp'],
                                                                                            shot_begin_timestamp=row['shot_begin_timestamp'],
                                                                                            distance=row['distance']))

    html_str += '</table></html>'
    return html_str


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
app.config['SECRET_KEY'] = b'\x13\rp\x96\xe3\x91\xd2\xac\x17\xc11\xb4y\xa5p\xa0'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# check whether the submitted variable is an integer
def allowed_num_results(num):
    try:
        int(num)
        return True
    except:
        logging.info('The submitted variable is not a valid number')
        return False


def md5sum(filename):
    with open(filename, mode='rb') as f:
        d = hashlib.md5()
        for buf in iter(partial(f.read, 128), b''):
            d.update(buf)
    return d.hexdigest()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        remove_letterbox = request.form.get('remove_letterbox')
        num_results = request.form.get('num_results')
        # if user does not select file, browser also
        # submits an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # if no number of results is submitted, a default of 30 is returned
        if num_results == '':
            num_results = 30
        if file and allowed_file(file.filename) and allowed_num_results(num_results):
            num_results = int(num_results)
            filename = secure_filename(file.filename)

            # save the uploaded image with a .. name and the correct file extension, to only ever save one image on disk
            filename = str(time.time()) + os.path.splitext(filename)[1]
            query_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(query_image_path):
                os.remove(query_image_path)

            file.save(query_image_path)
            query_image = Image.open(query_image_path)
            query_image = query_image.convert('RGB')
            buf = BytesIO()
            query_image.save(buf, 'PNG')
            query_image = buf.getvalue()
            query_image = base64.encodebytes(query_image).decode('ascii')

            url = 'http://{hostname}:9000/'.format(hostname=IMAGESEARCH_HOST)  # the name assigned in the docker subnet

            server_options = {}
            try:
                r = requests.get(url)
                if r.status_code == 200:
                    data = r.json()
                    caps = data.get('data', {}).get('capabilities', {})
                    for n in ('minimum_batch_size', 'maximum_batch_size', 'available_models'):
                        server_options[n] = caps.get(n, None)
            except requests.exceptions.RequestException:
                logging.info('try failed')
                pass

            # the response is what is sent to the server
            # Use a requests.session to use a KeepAlive connection to the server
            session = requests.session()
            headers = {"Content-Type": "application/json", "Accept": "application/json", "Cache-Control": "no-cache, no-store", "Pragma": "no-cache"}

            # call the main function of the query
            response = session.post(url, headers=headers, json={
                'query_image': query_image,
                'num_results': num_results,
                'remove_letterbox': remove_letterbox,
                'update_index': False
            })

            output = response.json()

            # get the results from the server as a list of dicts
            results = output.get('data')

            query_image_bytes = output.get('query_image_bytes')

            html_table = write_html_str(results, query_image_bytes)
            return html_table
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <textarea name="num_results"></textarea> 
      <input type=submit value=Upload>
      <label><input type="checkbox" name="remove_letterbox" value="True">remove letterbox</label>
    </form>
    '''


@app.route('/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=80)
