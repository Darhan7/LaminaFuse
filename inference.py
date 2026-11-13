"""Color fusion of already registered, normalized RGB and NIR inputs."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import torch
from models.lamina import Lamina

MODES = ('learned', 'analytic-contrast', 'analytic-intensity')

def validate(rgb, nir):
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or nir.shape != rgb.shape[:2]:
        raise ValueError('Expected RGB HxWx3 and registered NIR HxW on the same grid.')
    if min(nir.shape) < 32:
        raise ValueError('Images must be at least 32 pixels along each dimension.')
    if not all(np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1 for a in (rgb,nir)):
        raise ValueError('Inputs must be finite and explicitly normalized to [0,1].')

def compose_color(rgb, value):
    visible = rgb.max(-1)
    ratio = np.ones_like(rgb, dtype=np.float32)
    np.divide(rgb, visible[...,None], out=ratio, where=visible[...,None]>0)
    return np.clip(ratio * value[...,None], 0, 1)

def load_model(checkpoint, device):
    from models.weight_fusion_net import WeightFusionNet
    obj = torch.load(checkpoint, map_location='cpu', weights_only=True)
    kw = obj['fuser_kwargs']
    required = {'feat_dim':256,'reduction':4,'fuse_domain':'lamina','lamina_prior':True,
                'lamina_prior_p':2.0,'lamina_polarity':False,'fullres_refine':True,
                'dark_repair_enable':False,'use_guided_filter':False,'lifr_enable':False}
    if any(kw.get(k)!=v for k,v in required.items()) or kw.get('lifr2_enable',False):
        raise ValueError('Checkpoint configuration does not match the paper model.')
    model = WeightFusionNet(**kw).eval()
    model.load_state_dict(obj['fuser'], strict=True)
    return model.to(device)

@torch.inference_mode()
def fuse(rgb, nir, mode='analytic-contrast', *, model=None, device='cpu', valid=None):
    rgb = np.asarray(rgb, dtype=np.float32); nir = np.asarray(nir, dtype=np.float32)
    validate(rgb,nir)
    if mode not in MODES: raise ValueError(f'Unknown mode: {mode}')
    if valid is not None and np.asarray(valid).shape != nir.shape:
        raise ValueError('Validity mask must have the same shape as NIR.')
    r = torch.from_numpy(rgb.copy()).permute(2,0,1)[None].to(device)
    n = torch.from_numpy(nir.copy())[None,None].to(device)
    v = r.amax(1,keepdim=True)
    if mode == 'learned':
        if model is None: raise ValueError('Learned mode requires a loaded checkpoint.')
        result = model(r,n); value = result['f01']; weights = result['weights']
    else:
        lamina = Lamina(); z = lamina(v,n)
        logits = 2 * torch.cat([torch.log(z['e_v']+1e-12),torch.log(z['e_n']+1e-12)],1)
        weights = torch.softmax(logits,dim=1)
        if mode == 'analytic-contrast':
            value = lamina.reconstruct(z['base'],weights[:,0:1]*z['c_v']+weights[:,1:2]*z['c_n'])
        else:
            value = (weights[:,0:1]*v+weights[:,1:2]*n).clamp(0,1)
        value = (value*255).clamp(0,255)/255
    value = value[0,0].cpu().numpy()
    if valid is not None:value = np.where(valid,value,rgb.max(-1))
    return {'value':value,'color':compose_color(rgb,value),'weights':weights[0].cpu().numpy()}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rgb',required=True,type=Path);p.add_argument('--nir',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--mode',choices=MODES,default='analytic-contrast')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--valid-mask',type=Path)
    p.add_argument('--raw-output',type=Path,help='Optional float32 NPZ; use this for numerical evaluation.')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if a.mode=='learned' and a.checkpoint is None:p.error('--checkpoint is required for learned mode')
    if a.mode!='learned' and a.checkpoint is not None:p.error('--checkpoint only applies to learned mode')
    if a.threads<1:p.error('--threads must be positive')
    torch.set_num_threads(a.threads)
    with Image.open(a.rgb) as im:
        if im.mode!='RGB':p.error('RGB input must be an 8-bit RGB image; normalize other formats explicitly.')
        rgb=np.asarray(im,dtype=np.float32)/255
    with Image.open(a.nir) as im:
        if im.mode!='L':p.error('NIR input must be an 8-bit grayscale image; normalize other formats explicitly.')
        nir=np.asarray(im,dtype=np.float32)/255
    valid=None
    if a.valid_mask:
        with Image.open(a.valid_mask) as im:valid=np.asarray(im.convert('L'))>0
    model=load_model(a.checkpoint,a.device) if a.mode=='learned' else None
    out=fuse(rgb,nir,a.mode,model=model,device=a.device,valid=valid)
    if a.output.suffix.lower()!='.png':p.error('--output must be a PNG to avoid lossy output encoding')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.rint(out['color']*255).astype(np.uint8)).save(a.output)
    if a.raw_output:
        a.raw_output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.raw_output,**out)
    print(f'Wrote {a.output} ({a.mode}); no display enhancement applied.')
if __name__=='__main__': main()
