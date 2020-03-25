import librosa
import numpy as np
import torch
from absl import app
from absl import flags
from torch import autograd
from torch.utils.data import dataset

from dataset import output_helper
from model import pose_generator_norm

FLAGS = flags.FLAGS

flags.DEFINE_string('model',
                    '/home/lizhaoliu/repos/Music-Dance-Video-Synthesis/log'
                    '/model/best_generator_0400.pth',
                    'Path to trained model file.')
flags.DEFINE_string('audio', '/home/lizhaoliu/Downloads/sample.wav',
                    'Path to audio file.')
flags.DEFINE_string('output_dir', '/home/lizhaoliu/tmp/sample_wav',
                    'Path to audio file.')
flags.DEFINE_integer('frame_width', 360, 'Single frame width.')
flags.DEFINE_integer('frame_height', 640, 'Single frame height.')

_SR = 16000
_FPS = 10
_N_FRAMES_CHUNK = 50
_N_JOINTS = 18


class AudioDataset(dataset.Dataset):
    """Input audio dataset.

    The dataset each time emits an audio chunk of shape [frames_per_chunk, 1,
    samples_per_frame].
    """

    def __init__(self, audio_path) -> None:
        x, sr = librosa.load(audio_path, sr=None)
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


def main(args):
    del args

    # Loads model.
    g = pose_generator_norm.Generator(1)
    g.eval()
    g.load_state_dict(torch.load(FLAGS.model))
    g.cuda()

    # Loads input audio into dataset.
    ds = AudioDataset(FLAGS.audio)
    dataloader = torch.utils.data.DataLoader(ds,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=False)

    # Run inference.
    for i, audio_chunk in enumerate(dataloader):
        audio_chunk = autograd.Variable(
            audio_chunk.type(torch.cuda.FloatTensor))
        g_pose = g(audio_chunk)
        g_pose = g_pose.contiguous().cpu().detach().numpy()
        g_pose = g_pose.reshape((_N_FRAMES_CHUNK, _N_JOINTS, 2))

        g_pose[:, :, 0] = (g_pose[:, :, 0] + 1) * (FLAGS.frame_height // 2)
        g_pose[:, :, 1] = (g_pose[:, :, 1] + 1) * (FLAGS.frame_width // 2)
        g_pose = g_pose.astype(np.int)
        output_helper.save_batch_images_continuously(g_pose, i, FLAGS.output_dir)


if __name__ == '__main__':
    app.run(main)
