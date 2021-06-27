import numpy as np
import logging
from flask import Flask, request, jsonify, abort
import multiprocessing
from waitress import serve
import uuid
from functools import wraps
import io
from datetime import datetime
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import time
import ctypes
from torchvision import transforms

import redis
import traceback


# Create Flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# MT
pool = None


# Cache folders
root_cache_folder   = 'F:/SimFace/'
enroll_cache_folder = f'{root_cache_folder}/enroll'
find_cache_folder   = f'{root_cache_folder}/find'

# Redis
conn = redis.Redis()

# Init DL models
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')  # Windows HACK
torch.set_grad_enabled(False)
torch.set_num_threads(8)
mtcnn  = MTCNN(keep_all=True, device='cuda').eval().to('cuda')
resnet = InceptionResnetV1(pretrained='vggface2', device='cuda').eval().to('cuda')


# Create file app.logger and set formatter from Flask
formatter = app.logger.handlers[0].formatter

file_logger = logging.getLogger('Fileapp.logger')
file_logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('exec.log')
fh.setFormatter(formatter)

file_logger.addHandler(fh)

ip_ban_list = ['79.124.62.106', '185.176.222.39']

from flask_swagger import swagger
@app.route("/spec")
def spec():
    return jsonify(swagger(app))

@app.before_request
def block_method():
    ip = request.environ.get('REMOTE_ADDR')
    if ip in ip_ban_list:
        app.logger.info(f'Block {ip}')
        abort(403)

# FPS and Image/Sec printer
def print_perf(times, cnt):
    app.logger.info(f'Average single picture processing time: {times / cnt}, FPS {cnt / times}')


# Error app.logger alias
def error(message, print_stack=True):
    app.logger.error(message)
    if print_stack:
        print(traceback.format_exc())


# Nnet embedding cosine matcher
def cosine_match(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Fn to math embedding 1vsAll
# This fn just iterates over all Redis embeddings and
# filter only N closest matches
def find_from_em(target_em, top_n, target_id=""):
    full_probs = []

    # For each face in target photo
    max_cosine = 0
    for target in target_em:
        probs = {}

        # Get all redis embeddings
        keys = conn.keys()
        # For each embedding in Redis
        for key in keys:
            key = key.decode()
            # If target embedding skip it
            if key == target_id:
                continue

            # Get all embeddings for photo
            value = conn.get(key)
            value = np.frombuffer(value, dtype=np.float32).reshape((-1, 512))

            sub_res = []

            # For each face embedding in other photo
            for em in value:
                # Do match
                cosine = cosine_match(target, em)
                # Append result
                sub_res.append({'cos': str(cosine)})

            probs[key] = sub_res

        #print("PROBS", probs.values())
        # Python-way to code :)
        # Sort results such that
        # Max cosine distance from each photo as key to sort all results
        # And reverse for [0] - max, [-1] - min
        top_for_image = list(
            sorted(probs.items(), key=lambda item: float(max(item[1], key=lambda x: float(x['cos']))['cos']), reverse=True))


        # If number of matched results more then required just slice top_n
        if len(top_for_image) > top_n:
            top_for_image = top_for_image[:top_n]

        # Convert back to dict
        top_for_image = {k: v for k, v in top_for_image}

        # Memorize n best matches for face in target photo
        full_probs.append(top_for_image)

    return full_probs

def cache(folder, image, name):
    if folder:
        if name:
            cache_name = f'{name}_{uuid.uuid4().hex}'
        else:
            cache_name = f'{uuid.uuid4().hex}'

        image.save(f'{folder}/{cache_name}.jpg')
        return cache_name

def create_em_from_file(file, cache_path=None, cache_name=None, proba=0.96):
    # Read image from stream
    image = Image.open(file).convert('RGB')
    width, height = image.size

    new_height = 480
    new_width = int(new_height * width / height)

    if height > 480:
        image = image.resize((new_width, new_height), Image.ANTIALIAS)


    # Do face detection and align
    try:
        image_align_raw, probs = mtcnn(image, return_prob=True)
    except Exception as e:
        cache_name = cache(f'{cache_path}_bad', image, cache_name)
        app.logger.error(f'Failed to detect dace from {cache_name}, with e: {e}')
        return None

    # If no faces or biggest photo too small
    # return None
    image_align = []
    for i in range(len(probs)):
        if probs[i] is None or probs[i] < proba:
            continue

        image_align.append(image_align_raw[i])

    if len(image_align) == 0:
        cache_name = cache(f'{cache_path}_bad', image, cache_name)
        file_logger.info(f'{cache_name},{probs}')
        return None


    # Cache good results
    cache_name = cache(f'{cache_path}_good', image, cache_name)
    file_logger.info(f'{cache_name},{probs}')


    # Send align faces to GPU and extract face embeddings
    try:
        image_align = torch.stack(image_align).to('cuda')
        embedding = resnet(image_align).detach().cpu()
    except Exception as e:
        app.logger.error(f'Failed to get embedding from {cache_name}, with e: {e}')
        return None

    return embedding


# Decorator for tracing
def base_log(fn):
    @wraps(fn)
    def decorated_function(*args, **kwargs):
        start = datetime.now()

        app.logger.info(f'Start execute route "{fn.__name__}"')
        app.logger.info(f'From: {request.remote_addr}')

        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            error(f'While route {fn.__name__} error occurred, e: {e}')
            return {'status': 'bad'}

        end = datetime.now()
        app.logger.info(f'End execute route "{fn.__name__}", duration: {end - start}s')

        if result is None or result == "":
            return {'status': 'ok'}
        else:
            return {'status': 'ok', 'result': result}

    return decorated_function


@app.route('/size_db')
@base_log
# Get number of elements in Redis DB
def size_db():
    return {'size': conn.dbsize()}


@app.route('/dump_db')
@base_log
# Force dump Redis DB on drive
def dump_db():
    app.logger.info(f'Last redis save time {conn.lastsave()}')
    conn.bgsave()


@app.route('/clear')
@base_log
# Drop Redis DB
def clear():
    conn.flushdb()


def enroll_single_file(inputs):
    start = time.time()

    key, file = inputs
    print(start, key)
    embedding = create_em_from_file(
        file=file,
        cache_path=enroll_cache_folder,
        cache_name=key
    )

    # Write bad if cant create it
    if embedding is None:
        return key, 'bad', 0

    embedding = embedding.numpy().tobytes()

    # Add do Redis if all ok
    conn.set(key, embedding)

    return key, 'ok', time.time() - start



@app.route('/enroll_fast', methods=['POST'])
@base_log
# Add photo to search index
def enroll_fast():
    res = {}

    # Number of processed photos
    n_good = 1e-12  # Number of processed images
    n_total = 1e-12  # Number of total images
    times = 1e-12  # Total processed times

    print('a')

    print('b')
    enroll_results = pool.map(enroll_single_file, list(request.files.items()))
    print('c')
    for key, status, time in enroll_results:
        res[key] = status
        times += time
        if time != 0:
            n_good += 1

    n_total += len(enroll_results)

    # Log
    print_perf(times, n_good)

    # Fix for logging as int type
    n_total = n_total
    n_good = n_good

    app.logger.info(f'Total images: {n_total}, Processed {n_good}/{n_total}')
    file_logger.info(f'From {request.remote_addr}, enroll result: {res}')

    # Return results
    return res

@app.route('/enroll', methods=['POST'])
@base_log
# Add photo to search index
def enroll():
    res = {}

    # Number of processed photos
    n_good  = 1e-12  # Number of processed images
    n_total = 1e-12  # Number of total images
    times   = 1e-12  # Total processed times

    app.logger.info(f'Number of input files: {len(request.files)}')

    # For each photo
    for key, file in request.files.items():
        n_total += 1

        start = time.time()

        # Get embeddings
        embedding = create_em_from_file(
            file=file,
            cache_path=enroll_cache_folder,
            cache_name=key
        )

        # Write bad if cant create it
        if embedding is None:
            res[key] = 'bad'
            continue

        embedding = embedding.numpy().tobytes()

        # Add do Redis if all ok
        conn.set(key, embedding)
        res[key] = 'ok'

        # Increase cnt
        times  += time.time() - start
        n_good += 1


    # Log
    print_perf(times, n_good)

    # Fix for logging as int type
    n_total = n_total
    n_good  = n_good

    app.logger.info(f'Total images: {n_total}, Processed {n_good}/{n_total}')
    file_logger.info(f'From {request.remote_addr}, enroll result: {res}')

    # Return results
    return res


@app.route('/find_from_image')
@base_log
def find_from_image():
    # Get number N best matches

    top_n = int(request.args.get('top_n', 4))

    # Number of processed photos
    cnt   = 1e-12  # Very small number to avoid /0
    times = 1e-12  # Number of processed images

    results = {}

    # For each photo
    for key, file in request.files.items():
        start = time.time()

        # Get embeddings
        embedding = create_em_from_file(
            file,
            cache_path=find_cache_folder,
            cache_name=key,
            proba=0.7
        )

        # Write bad if cant create it
        if embedding is None:
            results[key] = 'bad'
            continue

        # Do matches
        result = find_from_em(embedding, top_n)
        results[key] = result

        # Increase cnt
        times += time.time() - start
        cnt += 1

    # Log
    print_perf(times, cnt)
    file_logger.info(
        f'From {request.remote_addr}, find_from_image result: {results}, from inputs: {request.files.keys()}'
        )

    # Return results
    return results


@app.route('/find_from_id')
@base_log
def find_from_id():
    # Get number N best matches

    d = request.get_json()
    top_n = d.get('top_n', 10)

    # Get embedding for ID
    target_id = d.get('id')
    target_em = conn.get(target_id)
    target_em = np.frombuffer(target_em, dtype=np.float32).reshape((-1, 512))

    # Do matches
    results = find_from_em(target_em, top_n, target_id)

    # Log
    file_logger.info(f'From {request.remote_addr}, find_from_id result: {results}, from inputs: {target_id}')

    # Return results
    return results


@app.route('/verify')
@base_log
def verify():
    # Get a, b ID's from json
    d = request.get_json()
    id1, id2 = d.get('id1'), d.get('id2')

    # Get embeddings from Redis
    em1 = conn.get(id1)
    em2 = conn.get(id2)

    em1 = np.frombuffer(em1, dtype=np.float32).reshape((-1, 512))
    em2 = np.frombuffer(em2, dtype=np.float32).reshape((-1, 512))

    # Do cosine match
    cosine = cosine_match(em1, em2)
    res = {'cos': str(cosine)}

    # Log
    file_logger.info(
        f'From {request.remote_addr}, verify result: {res}, from inputs: {id1}, {id2}'
    )

    # Return results
    return res


if __name__ == '__main__':
    #serve(app, host='0.0.0.0', port=5000, threads=32)
    app.run(host='0.0.0.0', port=5000)