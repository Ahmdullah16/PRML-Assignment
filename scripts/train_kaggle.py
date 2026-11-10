"""Resumable, validation-selected full-dataset CNN / AC-GAN experiment."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps
from prml import CLASSES, SIZE, LATENT
from prml.models import CNN, Discriminator, Generator
from prml.train import seed_everything, discriminator_loss, generator_loss, evaluate, write_json

EXPERIMENT = 'kaggle-four-class-seed42-v1'

class CachedImages(Dataset):
    """uint8 cache avoids repeated image decoding and limits RAM usage."""
    def __init__(self, root, rows, augment=False):
        self.rows, self.augment = rows, augment
        self.images = torch.empty((len(rows), 3, SIZE, SIZE), dtype=torch.uint8)
        for i,r in enumerate(rows):
            with Image.open(root/r['file']) as im:
                im = ImageOps.exif_transpose(im).convert('RGB').resize((SIZE,SIZE), Image.Resampling.BILINEAR)
                self.images[i] = torch.from_numpy(np.asarray(im).copy()).permute(2,0,1)
        self.labels = torch.tensor([r['label'] for r in rows])
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,i):
        x = self.images[i].float().div_(127.5).sub_(1)
        if self.augment and torch.rand(()) < .5:
            x = x.flip(2)
        return x, self.labels[i]

def snapshot(model, kind, epoch, manifest_hash):
    return dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, type=kind,
                classes=CLASSES, size=SIZE, seed=42, best_epoch=epoch, experiment=EXPERIMENT,
                split_manifest_sha256=manifest_hash)

def train(kind, trainset, valset, a, manifest_hash):
    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = (CNN() if kind=='cnn' else Discriminator()).to(device)
    gen = Generator().to(device) if kind=='gan' else None
    opt = torch.optim.AdamW(model.parameters(),lr=a.lr if kind=='cnn' else .0002,
                          betas=(.9,.999) if kind=='cnn' else (.5,.999), weight_decay=.0001 if kind=='cnn' else 0.)
    go = torch.optim.Adam(gen.parameters(), lr=.0002, betas=(.5,.999)) if gen else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=.5, min_lr=1e-5) if not gen else None
    shuffle_rng = torch.Generator().manual_seed(42)
    tl = DataLoader(trainset,batch_size=a.batch_size,shuffle=True,generator=shuffle_rng,num_workers=0)
    vl = DataLoader(valset,batch_size=a.batch_size,num_workers=0)
    out = a.output
    history, best, bad, start_epoch, elapsed = [], float('inf'), 0, 1, 0.
    resume = out/f'{kind}_resume.pt'
    if resume.exists():
        r = torch.load(resume,map_location=device,weights_only=False) # trusted locally generated optimizer/RNG state only
        assert r['manifest_hash'] == manifest_hash
        model.load_state_dict(r['model']); opt.load_state_dict(r['optimizer'])
        if gen:
            gen.load_state_dict(r['generator']); go.load_state_dict(r['g_optimizer'])
        if scheduler:
            scheduler.load_state_dict(r['scheduler'])
        history,best,bad,start_epoch,elapsed = r['history'],r['best'],r['bad'],r['epoch']+1,r['elapsed']
        torch.set_rng_state(r['rng'].cpu()); shuffle_rng.set_state(r['shuffle_rng'].cpu())
    started = time.perf_counter()
    budget = a.cnn_epochs if kind=='cnn' else a.gan_epochs
    for epoch in range(start_epoch,budget+1):
        model.train()
        if gen: gen.train()
        total=correct=0
        loss_sum=g_sum=0.
        component = {k:0. for k in ('d_real_source','d_fake_source','d_real_class','d_fake_class','g_source','g_class')}
        epoch_start=time.perf_counter()
        for batch_index,(x,y) in enumerate(tl):
            x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            if not gen:
                logits=model(x)
                loss=nn.functional.cross_entropy(logits,y)
            else:
                labels=torch.randint(0,4,(len(y),),device=device)
                fake=gen(torch.randn(len(y),LATENT,device=device),labels)
                real_out,fake_out=model(x),model(fake.detach())
                logits=real_out[1]
                loss=discriminator_loss(real_out,fake_out,y,labels)
                bce,ce=nn.functional.binary_cross_entropy_with_logits,nn.functional.cross_entropy
                vals=(bce(real_out[0],torch.ones_like(real_out[0])),bce(fake_out[0],torch.zeros_like(fake_out[0])),ce(real_out[1],y),ce(fake_out[1],labels))
                for key,value in zip(list(component)[:4],vals): component[key]+=value.item()*len(y)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite discriminator/CNN loss')
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite model gradients')
            nn.utils.clip_grad_norm_(model.parameters(),10.)
            opt.step()
            if gen:
                opt.zero_grad(set_to_none=True)
                for p in model.parameters(): p.requires_grad_(False)
                go.zero_grad(set_to_none=True)
                fo=model(fake)
                gl=generator_loss(fo,labels)
                if not torch.isfinite(gl): raise FloatingPointError('Nonfinite generator loss')
                gl.backward()
                grads=[p.grad for p in gen.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(torch.count_nonzero(g) for g in grads):
                    raise FloatingPointError('Generator gradients missing or invalid')
                nn.utils.clip_grad_norm_(gen.parameters(),10.)
                go.step()
                for p in model.parameters(): p.requires_grad_(True)
                g_sum+=gl.item()*len(y)
                component['g_source']+=bce(fo[0],torch.ones_like(fo[0])).item()*len(y)
                component['g_class']+=ce(fo[1],labels).item()*len(y)
            total+=len(y); correct+=(logits.argmax(1)==y).sum().item(); loss_sum+=loss.item()*len(y)
            if batch_index%100==0:
                print(f'{kind} epoch {epoch}/{budget} batch {batch_index+1}/{len(tl)} loss={loss.item():.4f}',flush=True)
        val=evaluate(model,vl,device,kind=='gan')
        row=dict(epoch=epoch,train_loss=loss_sum/total,train_accuracy=correct/total,
                 val_loss=val['loss'],val_accuracy=val['accuracy'],val_macro_f1=val['macro_f1'],
                 learning_rate=opt.param_groups[0]['lr'],seconds=time.perf_counter()-epoch_start)
        if gen:
            row.update(generator_loss=g_sum/total,**{k:v/total for k,v in component.items()})
        history.append(row)
        if val['loss'] < best-1e-5:
            best,bad=val['loss'],0
            torch.save(snapshot(model,kind,epoch,manifest_hash),out/f'{kind}.pt')
            if gen:
                torch.save(dict(model={k:v.detach().cpu().clone() for k,v in gen.state_dict().items()},
                    classes=CLASSES,size=SIZE,latent=LATENT,best_epoch=epoch,experiment=EXPERIMENT,
                    split_manifest_sha256=manifest_hash),out/'generator.pt')
        else: bad+=1
        if scheduler: scheduler.step(val['loss'])
        write_json(out/f'{kind}_history.json',history)
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),generator=gen.state_dict() if gen else None,
            g_optimizer=go.state_dict() if gen else None,scheduler=scheduler.state_dict() if scheduler else None,
            history=history,best=best,bad=bad,epoch=epoch,elapsed=elapsed+time.perf_counter()-started,
            manifest_hash=manifest_hash,rng=torch.get_rng_state(),shuffle_rng=shuffle_rng.get_state()),resume)
        print(json.dumps(dict(model=kind,**row)),flush=True)
        if bad>=a.patience:
            print(f'{kind}: early stopping after {bad} non-improving validation epochs',flush=True)
            break
    selected=torch.load(out/f'{kind}.pt',map_location='cpu',weights_only=True)
    metadata=dict(status='complete',epochs=len(history),best_epoch=selected['best_epoch'],
        best_validation_loss=best,training_seconds=elapsed+time.perf_counter()-started,
        parameters=sum(p.numel() for p in model.parameters()),generator_parameters=sum(p.numel() for p in gen.parameters()) if gen else 0,
        checkpoint_sha256=hashlib.sha256((out/f'{kind}.pt').read_bytes()).hexdigest())
    write_json(out/f'{kind}_training.json',metadata)
    return metadata

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/runs/kaggle_seed42'))
    p.add_argument('--manifest',type=Path,default=Path('results/kaggle_seed42/split_manifest.json'))
    p.add_argument('--cnn-epochs',type=int,default=20)
    p.add_argument('--gan-epochs',type=int,default=15)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--model',choices=['both','cnn','gan'],default='both')
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    a.output.mkdir(parents=True,exist_ok=True)
    manifest_hash=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    parts=json.loads(a.manifest.read_text())['partitions']
    config=dict(experiment=EXPERIMENT,seed=42,classes=list(CLASSES),size=SIZE,
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        split_manifest_sha256=manifest_hash,selection='minimum validation unweighted cross-entropy; early stopping patience 5',
        optimizer='CNN AdamW lr .001, weight decay .0001, betas .9/.999; GAN AdamW D weight decay 0 and Adam G, lr .0002, betas .5/.999',
        scheduler='CNN ReduceLROnPlateau validation CE, factor .5 patience 2; GAN fixed LR',
        hardware=dict(device='cuda' if torch.cuda.is_available() else 'cpu',cpu=platform.processor(),threads=a.threads),
        environment=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,platform=platform.platform()),
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path('prml/models.py'),Path('prml/train.py')]})
    write_json(a.output/'configuration.json',config)
    print('Caching train and validation images; test pixels are not loaded during training.',flush=True)
    trainset=CachedImages(a.data,parts['train'],True)
    valset=CachedImages(a.data,parts['validation'])
    for kind in ('cnn','gan') if a.model=='both' else (a.model,):
        if (a.output/f'{kind}_training.json').exists():
            print(f'{kind} completed run exists; preserving without retraining',flush=True)
            continue
        train(kind,trainset,valset,a,manifest_hash)

if __name__=='__main__': main()
