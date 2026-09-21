"""Run the existing rollout harness, also preserving lossless executed RGB frames."""
from pathlib import Path
import os
import runpy
import sys

import imageio
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
original_mimwrite = imageio.mimwrite


def lossless_and_video(uri, frames, *args, **kwargs):
    path = Path(uri)
    if path.parent.name == 'executed':
        array = np.asarray(frames, dtype=np.uint8)
        np.savez_compressed(path.with_suffix('.npz'), frames=array)
        directory = path.with_suffix('')
        directory.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(array):
            Image.fromarray(frame).save(directory/f'{i:04d}.png')
        print(f'LOSSLESS_ROLLOUT {path.with_suffix(".npz")} {array.shape}', flush=True)
    return original_mimwrite(uri, frames, *args, **kwargs)


imageio.mimwrite = lossless_and_video
sys.argv[0] = str(REPO/'eval/rollout.py')
runpy.run_path(sys.argv[0], run_name='__main__')
