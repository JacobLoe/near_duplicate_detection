import os
from flask import Flask, flash, request, redirect, url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename

import requests
from PIL import Image
from io import BytesIO
import base64
##########################################################################

UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def write_html_str(results, target_image):

    # write to html
    html_str = '<!DOCTYPE html><html lang="en"><table cellspacing="20"><tr><th>thumbnail</th><th>videofile</th><th>frame timestamp</th><th>shot_beginning</th><th>distance</th></tr>'
    # add the target image to the html
    html_str += str('<tr><td><img src="{}"></td></tr>'.format(target_image))

    # append the found images to the html
    for row in results:
        # write an entry for the table, format is: frame_path, source_video, frame_timestamp, shot_begin_frame, distance
        html_str += str('<tr><td><img src="{}"></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(row['frame_path'],
                                                                                                              row['source_video'],
                                                                                                              row['frame_timestamp'],
                                                                                                              row['shot_begin_frame'],
                                                                                                              row['distance']))

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
app.config['SECRET_KEY'] = 'ndd'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# check whether the submitted variable is an integer
def allowed_num_results(num):
    try:
        int(num)
        return True
    except:
        print('The submitted variable is not a valid number')
        return False


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        num_results = request.form.get('textbox')
        print(num_results)
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
            filename = ('target_image' + os.path.splitext(filename)[1])
            target_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(target_image_path)

            target_image = Image.open(target_image_path)
            buf = BytesIO()
            target_image.save(buf, 'PNG')
            target_image = buf.getvalue()
            target_image = base64.encodebytes(target_image).decode('ascii')

            url = 'http://localhost:9000/'

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
            response = session.post(url, headers=headers, json={
                'target_image': target_image,
                'num_results': num_results,
            })

            output = response.json()

            # get the results from the server as a list of dicts
            results = output.get('data')

            html_table = write_html_str(results, target_image_path)
            return html_table
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <textarea name="textbox"></textarea>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=80)
