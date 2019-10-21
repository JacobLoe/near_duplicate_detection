# pip3 install nose
# nosetests unit_letterbox.py
import pickle
import numpy as np
import multiprocessing as mp
import functools
from sklearn.metrics.pairwise import euclidean_distances
from extract_features import extract_features, load_model
from PIL import Image, ImageChops

inception_model = load_model()
pickle_features = 'features_pickle/features.pickle'
with open(pickle_features, 'rb') as handle:
    features_server, info_server = pickle.load(handle)
# archive_features = np.asarray(features_server['feature_list'], dtype='float16')
archive_features = np.asarray(features_server['feature_list'])
f_size = archive_features.shape[0]  # the amount of features
nproc = 4
batch_size = int(float(f_size) / nproc)
values_compared = 5


def compute_batch(start_idx, batch_size, Y):
    global archive_features
    dists = euclidean_distances(archive_features[start_idx:start_idx+batch_size, :], Y)
    return dists


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def test_letterbox_1():
    target_image_path = 'test_images/with_letterbox/201250.png'
    target_image = Image.open(target_image_path)
    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_image_no_letterbox = trim(target_image)

    target_feature = extract_features(inception_model, target_image)
    target_feature_no_letterbox = extract_features(inception_model, target_image_no_letterbox)

    compute_batch_ = functools.partial(compute_batch, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances = np.concatenate(distances)
    distances = np.reshape(distances, distances.shape[0])

    compute_batch_ = functools.partial(compute_batch, batch_size=batch_size, Y=target_feature_no_letterbox)
    with mp.Pool(nproc) as pool:
        distances_no_letterbox = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_no_letterbox = np.concatenate(distances_no_letterbox)
    distances_no_letterbox = np.reshape(distances_no_letterbox, distances_no_letterbox.shape[0])

    indices = np.argsort(distances).tolist()
    lowest_distances = [(distances[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    indices = np.argsort(distances_no_letterbox).tolist()
    lowest_distances_no_letterbox = [(distances_no_letterbox[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    p = []
    c = []
    for i_16, i_32 in zip(lowest_distances, lowest_distances_no_letterbox):
        a = (i_16[1] == i_32[1]) and (i_16[2] == i_32[2]) and (i_16[3] == i_32[3])
        p.append(a)
        c.append(True)

    assert p[:values_compared] == c[:values_compared]


if __name__ == "__main__":
    test_letterbox_1()