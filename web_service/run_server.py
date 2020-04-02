import sys

sys.path.append("../")

import flask
from model import pose_generator_norm
import torch
import time
import io
import json
import librosa
import os
import random
import numpy as np
import subprocess
import shutil
import base64
from torch.utils.data import dataset
from absl import app as absl_app
from absl import flags
from absl import logging
from concurrent import futures
from flask import request
from torch import autograd
from gevent import pywsgi
from dataset import output_helper

app = flask.Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
app._static_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  'static')

FLAGS = flags.FLAGS

flags.DEFINE_integer('port', 9876, 'Server port.')
flags.DEFINE_integer('frame_width', 360, 'Single frame width.')
flags.DEFINE_integer('frame_height', 640, 'Single frame height.')
flags.DEFINE_string('model', '', 'Path to trained model file.')
flags.DEFINE_string('ffmpeg_path', '/usr/bin/ffmpeg', 'FFMPEG binary path.')
flags.DEFINE_string('upload_folder', 'upload',
                    'Relative path to upload folder.')

_SR = 16000
_FPS = 10
_N_FRAMES_CHUNK = 50
_N_JOINTS = 18

_generator = None
_exec = futures.ThreadPoolExecutor()


class AudioDataset(dataset.Dataset):
    """Input audio dataset.

    The dataset each time emits an audio chunk of shape [frames_per_chunk, 1,
    samples_per_frame].
    """

    def __init__(self, audio_binary: bytes) -> None:
        args = (
            FLAGS.ffmpeg_path, '-i', '-', '-ac', '1', '-ar', str(_SR), '-f',
            'wav', '-')
        c = subprocess.run(args, input=audio_binary, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, shell=False)
        c.check_returncode()
        f = io.BytesIO(c.stdout)
        x, sr = librosa.load(f, sr=None)
        x = librosa.resample(x, sr, _SR)
        x = (x * (2 ** 15)).astype(np.int16)
        s_per_frame = _SR // _FPS
        n_frames = x.shape[0] // s_per_frame
        x = x[:(n_frames * s_per_frame)].reshape((n_frames, 1, s_per_frame))
        self.x = x

    def __getitem__(self, index: int):
        return self.x[index * _N_FRAMES_CHUNK:(index + 1) * _N_FRAMES_CHUNK]

    def __len__(self) -> int:
        return self.x.shape[0] // _N_FRAMES_CHUNK


def _create_pose_seq(audio_binary):
    """Creates pose sequence from given audio binary.

    Args:
        audio_binary: audio binary in bytes.

    Returns:
        A numpy array denoting the pose sequence, of shape (frames, 18, 2).
    """
    ds = AudioDataset(audio_binary)
    dataloader = torch.utils.data.DataLoader(ds,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=False)

    # Run inference.
    global _generator
    poses = []
    for i, audio_chunk in enumerate(dataloader):
        audio_chunk = autograd.Variable(
            audio_chunk.type(torch.cuda.FloatTensor))
        pose_chunk = _generator(audio_chunk)
        pose_chunk = pose_chunk.contiguous().cpu().detach().numpy()
        pose_chunk = pose_chunk.reshape((_N_FRAMES_CHUNK, _N_JOINTS, 2))
        poses.append(pose_chunk)
    pose_sequence = np.concatenate(poses, 0)
    return pose_sequence


@app.route('/dance', methods=['POST'])
def dance():
    if request.method == 'POST':
        start_time = time.time()
        # Loads input audio into dataset.
        audio_binary = request.get_data()
        if not audio_binary:
            logging.warning('Uploaded audio binary is empty.')
            flask.abort(400)
        pose_sequence = _create_pose_seq(audio_binary)
        elapsed_time = time.time() - start_time
        logging.info(
            f'Elapsed time: {elapsed_time:.2f} s, audio size: '
            f'{len(audio_binary)} B.')
        response = app.response_class(
            response=json.dumps({'pose_sequence': pose_sequence.tolist()}))
        return response


def _remove_after(filepath, delay):
    time.sleep(delay)
    os.remove((filepath))


def _read_file_chunk(fd, chunk_size=8192):
    while True:
        buf = fd.read(chunk_size)
        if buf:
            yield buf
        else:
            break


@app.route('/dance_figure', methods=['POST', 'GET'])
def dance_figure():
    if request.method == 'POST':
        f = request.files['uploaded_file']
        audio_binary = f.read()
        if not audio_binary:
            logging.warning('Uploaded audio binary is empty.')
            flask.abort(400)

        upload_dir = get_upload_dir()
        common_name = f'{time.strftime("%H_%M_%S")}_{random.randint(0, 1000)}'
        output_dir = os.path.join(upload_dir, common_name)
        v_file = f'{common_name}.mp4'
        output_video = os.path.join(os.path.dirname(__file__), 'static', v_file)
        pose_sequence = _create_pose_seq(audio_binary)
        pose_sequence[:, :, 0] = (pose_sequence[:, :, 0] + 1) * (
                FLAGS.frame_height // 2)
        pose_sequence[:, :, 1] = (pose_sequence[:, :, 1] + 1) * (
                FLAGS.frame_width // 2)
        pose_sequence = pose_sequence.astype(np.int)
        # Save pose sequence to image sequence.
        output_helper.save_batch_images_continuously(pose_sequence, 0,
                                                     output_dir)

        # Render a video using audio and image sequence.
        args = (
            FLAGS.ffmpeg_path,
            '-r', str(_FPS),
            '-i', os.path.join(output_dir, '%d.png'),
            '-i', '-',
            '-r', str(_FPS),
            '-y',
            output_video)
        c = subprocess.run(args, input=audio_binary, shell=False)
        c.check_returncode()
        with open(output_video, 'rb') as vf:
            vid_binary = vf.read()
            # Only base64 encoded binary be fed to HTML5 <video> element.
            b64_vid_binary = base64.b64encode(vid_binary)
            bb = io.BytesIO(b64_vid_binary)
        shutil.rmtree(output_dir)
        os.remove(output_video)

        return flask.Response(flask.stream_with_context(_read_file_chunk(bb)),
                              mimetype='video/mp4',
                              content_type='video/mp4',
                              direct_passthrough=True)


@app.route('/', methods=['GET'])
def index():
    """Index page."""
    return flask.render_template('index.html', video_width=FLAGS.frame_width,
                                 video_height=FLAGS.frame_height)


def get_upload_dir():
    base_path = os.path.dirname(__file__)
    upload_dir = os.path.join(base_path, FLAGS.upload_folder)
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    return upload_dir


def _init_model():
    """Loads and initializes PyTorch model."""
    logging.info(f"Loading model from '{FLAGS.model}'...")
    if not FLAGS.model:
        logging.fatal('Model path is not set.')
    global _generator
    _generator = pose_generator_norm.Generator(1)
    _generator.eval()
    _generator.load_state_dict(torch.load(FLAGS.model))
    _generator.cuda()


def main(args):
    del args

    _init_model()

    logging.info(f'Start server on :{FLAGS.port}...')
    http_server = pywsgi.WSGIServer(('', FLAGS.port), app)
    http_server.serve_forever()


if __name__ == '__main__':
    absl_app.run(main)
