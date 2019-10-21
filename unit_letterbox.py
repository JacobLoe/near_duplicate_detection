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
    target_image_no_letterbox = trim(target_image)  # remove the border from the image

    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_image_no_letterbox = target_image_no_letterbox.convert('RGB')
    target_image_no_letterbox = target_image_no_letterbox.resize((299, 299))


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
    for i, j in zip(lowest_distances, lowest_distances_no_letterbox):
        # print(i)
        # print(j)
        a = (i[1] == j[1]) and (i[2] == j[2]) and (i[3] == j[3])
        p.append(a)
        c.append(True)
    assert p[:values_compared] == c[:values_compared]


def test_letterbox_2():
    target_image_path = 'test_images/with_letterbox/362200.png'
    target_image = Image.open(target_image_path)
    target_image_no_letterbox = trim(target_image)  # remove the border from the image

    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_image_no_letterbox = target_image_no_letterbox.convert('RGB')
    target_image_no_letterbox = target_image_no_letterbox.resize((299, 299))


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
    for i, j in zip(lowest_distances[:values_compared], lowest_distances_no_letterbox[:values_compared]):
        # print(i)
        # print(j)
        a = (i[1] == j[1]) and (i[2] == j[2]) and (i[3] == j[3])
        p.append(a)
        c.append(True)
    assert p[:values_compared] == c[:values_compared]


def test_letterbox_3():
    target_image_path = 'test_images/with_letterbox/959040.png'
    target_image = Image.open(target_image_path)
    target_image_no_letterbox = trim(target_image)  # remove the border from the image

    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_image_no_letterbox = target_image_no_letterbox.convert('RGB')
    target_image_no_letterbox = target_image_no_letterbox.resize((299, 299))


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
    for i, j in zip(lowest_distances[:values_compared], lowest_distances_no_letterbox[:values_compared]):
        # print(i)
        # print(j)
        a = (i[1] == j[1]) and (i[2] == j[2]) and (i[3] == j[3])
        p.append(a)
        c.append(True)
    assert p[:values_compared] == c[:values_compared]


def test_letterbox_4():
    target_image_path = 'test_images/with_letterbox/1470083.png'
    target_image = Image.open(target_image_path)
    target_image_no_letterbox = trim(target_image)  # remove the border from the image

    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_image_no_letterbox = target_image_no_letterbox.convert('RGB')
    target_image_no_letterbox = target_image_no_letterbox.resize((299, 299))


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
    for i, j in zip(lowest_distances[:values_compared], lowest_distances_no_letterbox[:values_compared]):
        # print(i)
        # print(j)
        a = (i[1] == j[1]) and (i[2] == j[2]) and (i[3] == j[3])
        p.append(a)
        c.append(True)
    assert p[:values_compared] == c[:values_compared]


def test_letterbox_5():
    target_image_path = 'test_images/with_letterbox/6017320.png'
    target_image = Image.open(target_image_path)
    target_image_no_letterbox = trim(target_image)  # remove the border from the image

    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_image_no_letterbox = target_image_no_letterbox.convert('RGB')
    target_image_no_letterbox = target_image_no_letterbox.resize((299, 299))


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
    for i, j in zip(lowest_distances[:values_compared], lowest_distances_no_letterbox[:values_compared]):
        # print(i)
        # print(j)
        a = (i[1] == j[1]) and (i[2] == j[2]) and (i[3] == j[3])
        p.append(a)
        c.append(True)
    assert p[:values_compared] == c[:values_compared]


if __name__ == "__main__":
    test_letterbox_1()
