import librosa
import os
import glob
import numpy as np
import time


base_dir = '../../genres_wav/'
test_dir = '../../genres_test/'
genre_list = ['classical', 'jazz', 'pop', 'rock']


def shuffle_two(a, b):
    assert a.shape[0] == b.shape[0]
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]


def create_mfcc_file(path):
    y, sr = librosa.load(path, sr=44100)
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    m = np.mean(m, axis=1)
    mfcc_path = path[:-3] + 'mfcc'
    np.save(mfcc_path, m)


def read_mfcc_file(test=False, shuffle=False):
    dir = test_dir if test else base_dir
    X, y = [], []
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(dir, genre, "*.mfcc.npy")):
            mfcc = np.load(fn)
            X.append(mfcc)
            y.append(label)
    return shuffle_two(np.array(X), np.array(y)) if shuffle else (np.array(X), np.array(y))


if __name__ == '__main__':
    start = time.time()

    # Create MFCC file for training directory
    for currdir, subdirs, files in os.walk(base_dir):
        print(currdir, subdirs)
        files = [filename for filename in files if filename[-3:] == 'wav']
        for file in files:
            path = currdir + '/' + file
            create_mfcc_file(path)

    # Create MFCC file for test directory
    # for currdir, subdirs, files in os.walk(test_dir):
    #     print(currdir, subdirs)
    #     files = [filename for filename in files if filename[-3:] == 'wav']
    #     for file in files:
    #         path = currdir + '/' + file
    #         create_mfcc_file(path)

    # X, y = read_mfcc_file(dir=test_dir)
    # print(X.shape, y.shape)
    elapsed = time.time() - start
    print("Total time: {} s".format(elapsed))