""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """
""" modified by amaruak00 20200826"""

import os
import lmdb
import cv2

import numpy as np

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createDataset(input_path, output_path, _images, _idx=[], train=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        input_path  : input folder path where starts image_path
        output_path : LMDB output path
        _images : image list in input_path
        _idx : permutated list from image list
        train : if true, set cache as a train
    """
    os.makedirs(output_path, exist_ok=True)

    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    cnt = 1

    if train:
        size = int(np.ceil(len(_idx) * 0.8))
        idx = _idx[:size]
    else:
        size = int(np.ceil(len(_idx) * 0.2))
        idx = _idx[-size:]


    for i in idx:
        image_path = input_path + _images[i]
        label = _images[i].split('.')[0]

        if not os.path.exists(image_path):
            print('%s does not exist' % image_path)
            continue

        with open(image_path, 'rb') as f:
            image_bin = f.read()

        try:
            if not checkImageIsValid(image_bin):
                print('%s is not a valid image' % image_path)
                continue
        except:
            print('error occured', i)
            with open(output_path + '/error_image_log.txt', 'a') as log:
                log.write('%s-th image data occured error\n' % str(i))
            continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = image_bin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, size))
        cnt += 1

    size = cnt-1
    cache['num-samples'.encode()] = str(size).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % size)