import sys

sys.path.append("../")

import flask
from model import pose_generator_norm
import torch
import time
import io
import json
import librosa
import numpy as np
import subprocess
from torch.utils.data import dataset
from absl import app
from absl import flags
from absl import logging
from flask import request
from torch import autograd
from gevent import pywsgi

_app = flask.Flask(__name__)
_generator = None

FLAGS = flags.FLAGS

flags.DEFINE_integer('port', 9876, 'Server port')
flags.DEFINE_string('model', '', 'Path to trained model file.')

_SR = 16000
_FPS = 10
_N_FRAMES_CHUNK = 50
_N_JOINTS = 18


class AudioDataset(dataset.Dataset):
    """Input audio dataset.

    The dataset each time emits an audio chunk of shape [frames_per_chunk, 1,
    samples_per_frame].
    """

    def __init__(self, audio_binary: bytes) -> None:
        args = (
            '/usr/bin/ffmpeg', '-i', '-', '-ac', '1', '-ar', str(_SR), '-f',
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


@_app.route('/create_dance', methods=['POST'])
def create_dance():
    if request.method == 'POST':
        start_time = time.time()
        # Loads input audio into dataset.
        audio_binary = request.get_data()
        if not audio_binary:
            flask.abort(400)
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
        pose_sequence.tolist()
        elapsed_time = time.time() - start_time
        logging.info(
            f'Elapsed time: {elapsed_time:.2f} s, audio size: '
            f'{len(audio_binary)} B.')
        response = _app.response_class(
            response=json.dumps({'pose_sequence': pose_sequence.tolist()}))
        return response


def main(args):
    del args

    logging.info(f"Loading model from '{FLAGS.model}'...")
    if not FLAGS.model:
        logging.fatal('Model path is not set.')
    global _generator
    _generator = pose_generator_norm.Generator(1)
    _generator.eval()
    _generator.load_state_dict(torch.load(FLAGS.model))
    _generator.cuda()

    logging.info(f'Start server on :{FLAGS.port}...')
    http_server = pywsgi.WSGIServer(('', FLAGS.port), _app)
    http_server.serve_forever()


if __name__ == '__main__':
    app.run(main)
