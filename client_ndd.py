import requests
from PIL import Image
from io import BytesIO
import base64
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("target_image", help="path to the target image")
    args = parser.parse_args()

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

    target_image = args.target_image
    num_results = 30
    num_cores = 8

    target_image = Image.open(target_image)
    buf = BytesIO()
    target_image.save(buf, 'PNG')
    target_image = buf.getvalue()
    target_image = base64.encodebytes(target_image).decode('ascii')

    response = session.post(url, headers=headers, json={
        'target_image': target_image,
        'num_results': num_results,
        'num_cores': num_cores
    })

    output = response.json()

    # get the results from the server
    concepts = output.get('data')
