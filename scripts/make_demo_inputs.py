"""Create synthetic inputs for an installation check, not experimental evidence."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=Path('outputs/demo'));a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
y,x=np.mgrid[:96,:160];v=(.10+.22*x/159).astype(np.float32);v[40:55,30:130]=.65
rgb=v[...,None]*np.array([1,.82,.55],np.float32);nir=(.30+.4*x/159).astype(np.float32);nir[25:70,20:140]=1
Image.fromarray(np.rint(rgb*255).astype('uint8')).save(a.output_dir/'rgb.png')
Image.fromarray(np.rint(nir*255).astype('uint8')).save(a.output_dir/'nir.png')
print('Synthetic inputs written to',a.output_dir)
