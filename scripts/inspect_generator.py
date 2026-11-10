"""Save a private fixed-seed sample grid from the paired selected generator.

Not a perceptual-quality or memorization test. Never exports source photographs.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from prml import CLASSES,LATENT
from prml.models import Generator

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=Path('results/runs/kaggle_seed42'))
    p.add_argument('--output',type=Path,default=Path('results/private/kaggle_seed42/generated_samples.png'))
    a=p.parse_args(); torch.set_num_threads(2); torch.manual_seed(42)
    g=torch.load(a.run/'generator.pt',map_location='cpu',weights_only=True)
    d=torch.load(a.run/'gan.pt',map_location='cpu',weights_only=True)
    assert g['best_epoch']==d['best_epoch'] and g['split_manifest_sha256']==d['split_manifest_sha256']
    model=Generator(); model.load_state_dict(g['model']); model.eval()
    labels=torch.arange(4).repeat_interleave(8)
    with torch.inference_mode():
        samples=model(torch.randn(len(labels),LATENT),labels)
    assert samples.shape==(32,3,64,64) and torch.isfinite(samples).all()
    values=((samples.clamp(-1,1)+1)*127.5).byte().permute(0,2,3,1).numpy()
    canvas=Image.new('RGB',(8*96,4*122),'white'); draw=ImageDraw.Draw(canvas)
    for i,value in enumerate(values):
        x=(i%8)*96; y=(i//8)*122
        canvas.paste(Image.fromarray(value).resize((96,96)),(x,y+24))
        if i%8==0: draw.text((x+3,y+5),CLASSES[int(labels[i])].title(),fill='black')
    a.output.parent.mkdir(parents=True,exist_ok=True); canvas.save(a.output)
    print(f'Private samples saved at selected epoch {g["best_epoch"]}; not published',flush=True)

if __name__=='__main__': main()
