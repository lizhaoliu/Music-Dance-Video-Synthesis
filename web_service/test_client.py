from absl import app
import requests
import numpy as np
import subprocess

def main(args):
    del args

    with open('/home/lizhaoliu/Downloads/SoundHelix-Song-1.mp3', 'rb') as f:
        audio = f.read()
    resp = requests.post('http://localhost:9876/dance', data=audio)
    pose_seq = np.array(resp.json()['pose_sequence'])
    print(pose_seq.shape)

if __name__ == '__main__':
    app.run(main)
