# pip3 install nose
# nosetests unit_cast.py
import pickle
import numpy as np
import multiprocessing as mp
import functools
from sklearn.metrics.pairwise import euclidean_distances
from extract_features import extract_features, load_model
from PIL import Image

inception_model = load_model()
pickle_features = 'features_pickle/features.pickle'
with open(pickle_features, 'rb') as handle:
    features_server, info_server = pickle.load(handle)
archive_features_16 = np.asarray(features_server['feature_list'], dtype='float16')
archive_features_32 = np.asarray(features_server['feature_list'])
f_size = archive_features_16.shape[0]  # the amount of features
nproc = 4
batch_size = int(float(f_size) / nproc)
values_compared = 40


def compute_batch_16(start_idx, batch_size, Y):
    global archive_features_16
    dists = euclidean_distances(archive_features_16[start_idx:start_idx+batch_size, :], Y)
    return dists


def compute_batch_32(start_idx, batch_size, Y):
    global archive_features_32
    dists = euclidean_distances(archive_features_32[start_idx:start_idx+batch_size, :], Y)
    return dists


def test_cast_1():
    target_image_path = 'test_images/211176.png'
    target_image = Image.open(target_image_path)
    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_feature = extract_features(inception_model, target_image)

    compute_batch_ = functools.partial(compute_batch_16, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_16 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_16 = np.concatenate(distances_16)
    distances_16 = np.reshape(distances_16, distances_16.shape[0])

    compute_batch_ = functools.partial(compute_batch_32, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_32 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_32 = np.concatenate(distances_32)
    distances_32 = np.reshape(distances_32, distances_32.shape[0])

    indices = np.argsort(distances_16).tolist()
    lowest_distances_16 = [(distances_16[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    indices = np.argsort(distances_32).tolist()
    lowest_distances_32 = [(distances_32[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    p = []
    c = []
    for i_16, i_32 in zip(lowest_distances_16, lowest_distances_32):
        a = (i_16[1] == i_32[1]) and (i_16[2] == i_32[2]) and (i_16[3] == i_32[3])
        p.append(a)
        c.append(True)

    print(lowest_distances_16[:values_compared])
    print(lowest_distances_32[:values_compared])
    assert p[:values_compared] == c[:values_compared]


def test_cast_2():
    target_image_path = 'test_images/image2.jpg'
    target_image = Image.open(target_image_path)
    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_feature = extract_features(inception_model, target_image)

    compute_batch_ = functools.partial(compute_batch_16, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_16 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_16 = np.concatenate(distances_16)
    distances_16 = np.reshape(distances_16, distances_16.shape[0])

    compute_batch_ = functools.partial(compute_batch_32, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_32 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_32 = np.concatenate(distances_32)
    distances_32 = np.reshape(distances_32, distances_32.shape[0])

    indices = np.argsort(distances_16).tolist()
    lowest_distances_16 = [(distances_16[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    indices = np.argsort(distances_32).tolist()
    lowest_distances_32 = [(distances_32[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    p = []
    c = []
    for i_16, i_32 in zip(lowest_distances_16, lowest_distances_32):
        a = (i_16[1] == i_32[1]) and (i_16[2] == i_32[2]) and (i_16[3] == i_32[3])
        p.append(a)
        c.append(True)

    print(lowest_distances_16[:values_compared])
    print(lowest_distances_32[:values_compared])
    assert p[:values_compared] == c[:values_compared]


def test_cast_3():
    target_image_path = 'test_images/image13.jpg'
    target_image = Image.open(target_image_path)
    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_feature = extract_features(inception_model, target_image)

    compute_batch_ = functools.partial(compute_batch_16, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_16 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_16 = np.concatenate(distances_16)
    distances_16 = np.reshape(distances_16, distances_16.shape[0])

    compute_batch_ = functools.partial(compute_batch_32, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_32 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_32 = np.concatenate(distances_32)
    distances_32 = np.reshape(distances_32, distances_32.shape[0])

    indices = np.argsort(distances_16).tolist()
    lowest_distances_16 = [(distances_16[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    indices = np.argsort(distances_32).tolist()
    lowest_distances_32 = [(distances_32[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    p = []
    c = []
    for i_16, i_32 in zip(lowest_distances_16, lowest_distances_32):
        a = (i_16[1] == i_32[1]) and (i_16[2] == i_32[2]) and (i_16[3] == i_32[3])
        p.append(a)
        c.append(True)

    print(lowest_distances_16[:values_compared])
    print(lowest_distances_32[:values_compared])
    assert p[:values_compared] == c[:values_compared]


def test_cast_4():
    target_image_path = 'test_images/Screenshot_David_Ed.png'
    target_image = Image.open(target_image_path)
    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_feature = extract_features(inception_model, target_image)

    compute_batch_ = functools.partial(compute_batch_16, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_16 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_16 = np.concatenate(distances_16)
    distances_16 = np.reshape(distances_16, distances_16.shape[0])

    compute_batch_ = functools.partial(compute_batch_32, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_32 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_32 = np.concatenate(distances_32)
    distances_32 = np.reshape(distances_32, distances_32.shape[0])

    indices = np.argsort(distances_16).tolist()
    lowest_distances_16 = [(distances_16[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    indices = np.argsort(distances_32).tolist()
    lowest_distances_32 = [(distances_32[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    p = []
    c = []
    for i_16, i_32 in zip(lowest_distances_16, lowest_distances_32):
        a = (i_16[1] == i_32[1]) and (i_16[2] == i_32[2]) and (i_16[3] == i_32[3])
        p.append(a)
        c.append(True)

    print(lowest_distances_16[:values_compared])
    print(lowest_distances_32[:values_compared])
    assert p[:values_compared] == c[:values_compared]


def test_cast_5():
    target_image_path = 'test_images/ts_broker_160908_0440.jpg'
    target_image = Image.open(target_image_path)
    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_feature = extract_features(inception_model, target_image)

    compute_batch_ = functools.partial(compute_batch_16, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_16 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_16 = np.concatenate(distances_16)
    distances_16 = np.reshape(distances_16, distances_16.shape[0])

    compute_batch_ = functools.partial(compute_batch_32, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_32 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_32 = np.concatenate(distances_32)
    distances_32 = np.reshape(distances_32, distances_32.shape[0])

    indices = np.argsort(distances_16).tolist()
    lowest_distances_16 = [(distances_16[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    indices = np.argsort(distances_32).tolist()
    lowest_distances_32 = [(distances_32[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    p = []
    c = []
    for i_16, i_32 in zip(lowest_distances_16, lowest_distances_32):
        a = (i_16[1] == i_32[1]) and (i_16[2] == i_32[2]) and (i_16[3] == i_32[3])
        p.append(a)
        c.append(True)

    print(lowest_distances_16[:values_compared])
    print(lowest_distances_32[:values_compared])
    assert p[:values_compared] == c[:values_compared]


def test_cast_6():
    target_image_path = 'test_images/ts_broker_160908_0440 (copy).jpg'
    target_image = Image.open(target_image_path)
    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_feature = extract_features(inception_model, target_image)

    compute_batch_ = functools.partial(compute_batch_16, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_16 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_16 = np.concatenate(distances_16)
    distances_16 = np.reshape(distances_16, distances_16.shape[0])

    compute_batch_ = functools.partial(compute_batch_32, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_32 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_32 = np.concatenate(distances_32)
    distances_32 = np.reshape(distances_32, distances_32.shape[0])

    indices = np.argsort(distances_16).tolist()
    lowest_distances_16 = [(distances_16[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    indices = np.argsort(distances_32).tolist()
    lowest_distances_32 = [(distances_32[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    p = []
    c = []
    for i_16, i_32 in zip(lowest_distances_16, lowest_distances_32):
        a = (i_16[1] == i_32[1]) and (i_16[2] == i_32[2]) and (i_16[3] == i_32[3])
        p.append(a)
        c.append(True)

    print(lowest_distances_16[:values_compared])
    print(lowest_distances_32[:values_compared])
    assert p[:values_compared] == c[:values_compared]


def test_cast_7():
    target_image_path = 'test_images/ts_kfw_170908_0408.png'
    target_image = Image.open(target_image_path)
    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_feature = extract_features(inception_model, target_image)

    compute_batch_ = functools.partial(compute_batch_16, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_16 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_16 = np.concatenate(distances_16)
    distances_16 = np.reshape(distances_16, distances_16.shape[0])

    compute_batch_ = functools.partial(compute_batch_32, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_32 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_32 = np.concatenate(distances_32)
    distances_32 = np.reshape(distances_32, distances_32.shape[0])

    indices = np.argsort(distances_16).tolist()
    lowest_distances_16 = [(distances_16[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    indices = np.argsort(distances_32).tolist()
    lowest_distances_32 = [(distances_32[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    p = []
    c = []
    for i_16, i_32 in zip(lowest_distances_16, lowest_distances_32):
        a = (i_16[1] == i_32[1]) and (i_16[2] == i_32[2]) and (i_16[3] == i_32[3])
        p.append(a)
        c.append(True)

    print(lowest_distances_16[:values_compared])
    print(lowest_distances_32[:values_compared])
    assert p[:values_compared] == c[:values_compared]


def test_cast_8():
    target_image_path = 'test_images/ts_kfw_170908_0408.png'
    target_image = Image.open(target_image_path)
    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_feature = extract_features(inception_model, target_image)

    compute_batch_ = functools.partial(compute_batch_16, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_16 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_16 = np.concatenate(distances_16)
    distances_16 = np.reshape(distances_16, distances_16.shape[0])

    compute_batch_ = functools.partial(compute_batch_32, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_32 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_32 = np.concatenate(distances_32)
    distances_32 = np.reshape(distances_32, distances_32.shape[0])

    indices = np.argsort(distances_16).tolist()
    lowest_distances_16 = [(distances_16[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    indices = np.argsort(distances_32).tolist()
    lowest_distances_32 = [(distances_32[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    p = []
    c = []
    for i_16, i_32 in zip(lowest_distances_16, lowest_distances_32):
        a = (i_16[1] == i_32[1]) and (i_16[2] == i_32[2]) and (i_16[3] == i_32[3])
        p.append(a)
        c.append(True)

    print(lowest_distances_16[:values_compared])
    print(lowest_distances_32[:values_compared])
    assert p[:values_compared] == c[:values_compared]


def test_cast_9():
    target_image_path = 'test_images/ts_kfw_170908_0408 (copy).png'
    target_image = Image.open(target_image_path)
    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_feature = extract_features(inception_model, target_image)

    compute_batch_ = functools.partial(compute_batch_16, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_16 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_16 = np.concatenate(distances_16)
    distances_16 = np.reshape(distances_16, distances_16.shape[0])

    compute_batch_ = functools.partial(compute_batch_32, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_32 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_32 = np.concatenate(distances_32)
    distances_32 = np.reshape(distances_32, distances_32.shape[0])

    indices = np.argsort(distances_16).tolist()
    lowest_distances_16 = [(distances_16[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    indices = np.argsort(distances_32).tolist()
    lowest_distances_32 = [(distances_32[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    p = []
    c = []
    for i_16, i_32 in zip(lowest_distances_16, lowest_distances_32):
        a = (i_16[1] == i_32[1]) and (i_16[2] == i_32[2]) and (i_16[3] == i_32[3])
        p.append(a)
        c.append(True)

    print(lowest_distances_16[:values_compared])
    print(lowest_distances_32[:values_compared])
    assert p[:values_compared] == c[:values_compared]


def test_cast_10():
    target_image_path = 'test_images/ts_nyse_150908_0123.png'
    target_image = Image.open(target_image_path)
    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_feature = extract_features(inception_model, target_image)

    compute_batch_ = functools.partial(compute_batch_16, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_16 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_16 = np.concatenate(distances_16)
    distances_16 = np.reshape(distances_16, distances_16.shape[0])

    compute_batch_ = functools.partial(compute_batch_32, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_32 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_32 = np.concatenate(distances_32)
    distances_32 = np.reshape(distances_32, distances_32.shape[0])

    indices = np.argsort(distances_16).tolist()
    lowest_distances_16 = [(distances_16[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    indices = np.argsort(distances_32).tolist()
    lowest_distances_32 = [(distances_32[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    p = []
    c = []
    for i_16, i_32 in zip(lowest_distances_16, lowest_distances_32):
        a = (i_16[1] == i_32[1]) and (i_16[2] == i_32[2]) and (i_16[3] == i_32[3])
        p.append(a)
        c.append(True)

    print(lowest_distances_16[:values_compared])
    print(lowest_distances_32[:values_compared])
    assert p[:values_compared] == c[:values_compared]


def test_cast_11():
    target_image_path = 'test_images/ts_nyse_150908_0123 (copy).png'
    target_image = Image.open(target_image_path)
    target_image = target_image.convert('RGB')
    target_image = target_image.resize((299, 299))
    target_feature = extract_features(inception_model, target_image)

    compute_batch_ = functools.partial(compute_batch_16, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_16 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_16 = np.concatenate(distances_16)
    distances_16 = np.reshape(distances_16, distances_16.shape[0])

    compute_batch_ = functools.partial(compute_batch_32, batch_size=batch_size, Y=target_feature)
    with mp.Pool(nproc) as pool:
        distances_32 = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))
    distances_32 = np.concatenate(distances_32)
    distances_32 = np.reshape(distances_32, distances_32.shape[0])

    indices = np.argsort(distances_16).tolist()
    lowest_distances_16 = [(distances_16[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    indices = np.argsort(distances_32).tolist()
    lowest_distances_32 = [(distances_32[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]

    p = []
    c = []
    for i_16, i_32 in zip(lowest_distances_16, lowest_distances_32):
        a = (i_16[1] == i_32[1]) and (i_16[2] == i_32[2]) and (i_16[3] == i_32[3])
        p.append(a)
        c.append(True)

    print(lowest_distances_16[:values_compared])
    print(lowest_distances_32[:values_compared])
    assert p[:values_compared] == c[:values_compared]


if __name__ == "__main__":
    test_cast_1()
