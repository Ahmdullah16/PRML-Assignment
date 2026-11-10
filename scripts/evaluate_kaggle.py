"""Final held-out evaluation, saved predictions, metrics and publication figures."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from prml import CLASSES
from prml.inference import load_checkpoint, predict
from prml.train import evaluate, write_json
from prml.metrics import classification_metrics
from scripts.train_kaggle import CachedImages, EXPERIMENT

DISPLAY = {'happiness':'Happiness','sadness':'Sadness','fear':'Fear','neutral':'Neutral'}
ORDER = [1,0,3,2]

def csv_write(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def figures(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'savefig.dpi':240,'axes.spines.top':False,'axes.spines.right':False})
    plots=out/'figures'; plots.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout(); fig.savefig(plots/name,bbox_inches='tight'); plt.close(fig)
    for kind in ('cnn','gan'):
        name=kind.upper()
        cm=np.array(summary['models'][kind]['test']['confusion_matrix'])[np.ix_(ORDER,ORDER)]
        for normalized in (False,True):
            values=cm/cm.sum(1,keepdims=True) if normalized else cm
            fig,ax=plt.subplots(figsize=(6.5,5.5))
            im=ax.imshow(values,cmap='Blues',vmin=0,vmax=1 if normalized else None)
            ax.set(xticks=range(4),yticks=range(4),xticklabels=[DISPLAY[CLASSES[i]] for i in ORDER],
                yticklabels=[DISPLAY[CLASSES[i]] for i in ORDER],xlabel='Predicted expression',ylabel='Actual expression',
                title=f'{name} test confusion matrix'+(' (row normalized)' if normalized else ' (counts)'))
            for i in range(4):
                for j in range(4):
                    ax.text(j,i,f'{values[i,j]:.1%}' if normalized else str(values[i,j]),ha='center',va='center',
                        color='white' if values[i,j]>values.max()*.5 else '#172d40')
            fig.colorbar(im,ax=ax,label='Proportion' if normalized else 'Images')
            save(fig,f'{name}_Confusion_Matrix'+('_Normalized' if normalized else '')+'.png')
        hist=json.loads((out/f'{kind}_history.json').read_text())
        epochs=[r['epoch'] for r in hist]
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot(epochs,[r['train_accuracy'] for r in hist],label='Training (augmented minibatches)')
        ax.plot(epochs,[r['val_accuracy'] for r in hist],label='Validation (no augmentation)')
        ax.axvline(summary['models'][kind]['best_epoch'],color='gray',ls=':',label='Selected epoch')
        ax.set(xlabel='Epoch',ylabel='Accuracy',ylim=(0,1),title=f'{name} classification accuracy'); ax.legend(fontsize=8)
        save(fig,'CNN_Training_Accuracy.png' if kind=='cnn' else 'GAN_Classification_Accuracy.png')
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot(epochs,[r['train_loss'] for r in hist],label='CNN training CE' if kind=='cnn' else 'Discriminator total objective')
        if kind=='cnn': ax.plot(epochs,[r['val_loss'] for r in hist],label='Validation classification CE')
        else: ax.plot(epochs,[r['generator_loss'] for r in hist],label='Generator total objective')
        ax.set(xlabel='Epoch',ylabel='Mean loss',title=f'{name} training objectives'); ax.legend(fontsize=8)
        save(fig,f'{name}_Training_Loss.png')
    for key,name in [('accuracy','Accuracy'),('macro_precision','Precision'),('macro_recall','Recall'),('macro_f1','F1_Score')]:
        fig,ax=plt.subplots(figsize=(5.5,4))
        values=[summary['models'][k]['test'][key] for k in ('cnn','gan')]
        bars=ax.bar(['CNN','AC-GAN classifier'],values,color=['#167e87','#b65b43'],width=.55)
        ax.bar_label(bars,labels=[f'{v:.2%}' for v in values],padding=5)
        ax.set(ylim=(0,1),ylabel='Score',title=('Test '+name.replace('_',' ')+(' (macro)' if key!='accuracy' else '')))
        save(fig,name+'_Comparison.png')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--run',type=Path,default=Path('results/runs/kaggle_seed42'))
    p.add_argument('--output',type=Path,default=Path('results/kaggle_seed42'))
    a=p.parse_args(); torch.set_num_threads(4)
    manifest_path=a.output/'split_manifest.json'; digest=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    parts=json.loads(manifest_path.read_text())['partitions']
    # Gate pixel access on completion of both validation-selected training runs.
    assert all((a.run/f'{k}_training.json').exists() for k in ('cnn','gan'))
    for row in parts['test']:
        if hashlib.sha256((a.data/row['file']).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError(f"Held-out input hash changed: {row['file']}")
    loader=DataLoader(CachedImages(a.data,parts['test']),batch_size=64,shuffle=False)
    summary=dict(experiment=EXPERIMENT,status='complete',labels=list(CLASSES),split_manifest_sha256=digest,
                 configuration=json.loads((a.run/'configuration.json').read_text()),models={})
    predictions=[]
    majority=max(range(4),key=lambda c:sum(r['label']==c for r in parts['train']))
    summary['training_majority_baseline']=dict(selected_class=CLASSES[majority],
        **classification_metrics([r['label'] for r in parts['test']],[majority]*len(parts['test'])))
    for kind in ('cnn','gan'):
        net,meta=load_checkpoint(a.run/f'{kind}.pt')
        assert meta['split_manifest_sha256']==digest and meta['experiment']==EXPERIMENT
        measured=evaluate(net,loader,'cpu',kind=='gan')
        actual,pred,probs=[measured.pop(k) for k in ('actual','predicted','probabilities')]
        report=classification_report(actual,pred,labels=list(range(4)),target_names=CLASSES,output_dict=True,zero_division=0)
        assert np.array_equal(confusion_matrix(actual,pred,labels=list(range(4))),measured['confusion_matrix'])
        assert np.isclose(accuracy_score(actual,pred),measured['accuracy'])
        for key,metric in [('macro_precision','precision'),('macro_recall','recall'),('macro_f1','f1-score')]:
            assert np.isclose(measured[key],report['macro avg'][metric])
        measured.update({f'weighted_{k}':report['weighted avg'][v] for k,v in [('precision','precision'),('recall','recall'),('f1','f1-score')]})
        for row,y,yp,prob in zip(parts['test'],actual,pred,probs):
            predictions.append(dict(model=kind,file=row['file'],sha256=row['sha256'],actual=CLASSES[y],predicted=CLASSES[yp],
                **{f'p_{c}':p for c,p in zip(CLASSES,prob)}))
        training=json.loads((a.run/f'{kind}_training.json').read_text())
        summary['models'][kind]=training|{'test':measured,'classification_report':report}
        csv_write(a.output/f'{kind.upper()}_Metrics.csv',[dict(metric=k,value=measured[k]) for k in
            ('accuracy','balanced_accuracy','macro_precision','macro_recall','macro_f1','weighted_precision','weighted_recall','weighted_f1')])
        csv_write(a.output/f'Classification_Report_{kind.upper()}.csv',[dict(expression=c,**report[c]) for c in CLASSES])
        shutil.copyfile(a.run/f'{kind}_history.json',a.output/f'{kind}_history.json')
    csv_write(a.output/'Test_Predictions.csv',predictions)
    csv_write(a.output/'Model_Comparison.csv',[dict(model=k,**{m:v['test'][m] for m in
        ('accuracy','macro_precision','macro_recall','macro_f1','balanced_accuracy','weighted_f1')},
        training_seconds=v['training_seconds'],parameters=v['parameters'],best_epoch=v['best_epoch']) for k,v in summary['models'].items()])
    write_json(a.output/'Experiment_Configuration.json',summary['configuration'])
    write_json(a.output/'summary.json',summary)
    figures(a.output,summary)
    # Replay from disk-saved probabilities and labels independently.
    with (a.output/'Test_Predictions.csv').open(newline='') as f: saved=list(csv.DictReader(f))
    for k in ('cnn','gan'):
        rows=[r for r in saved if r['model']==k]
        assert len(rows)==len(parts['test'])
        cm=confusion_matrix([CLASSES.index(r['actual']) for r in rows],[CLASSES.index(r['predicted']) for r in rows],labels=range(4))
        assert cm.sum()==len(parts['test']) and np.array_equal(cm,summary['models'][k]['test']['confusion_matrix'])
        # Check actual uploaded-image preprocessing and inference on representative held-out files.
        net,_=load_checkpoint(a.run/f'{k}.pt')
        for label in range(4):
            row=next(r for r in rows if CLASSES.index(r['actual'])==label)
            scores=predict(net,k,a.data/row['file'])
            assert tuple(scores)==CLASSES and np.isclose(sum(scores.values()),1)
            assert np.allclose(list(scores.values()),[float(row[f'p_{c}']) for c in CLASSES],atol=1e-5)
    write_json(a.output/'verification.json',dict(saved_predictions_recomputed=True,sklearn_metrics_crosschecked=True,
        checkpoint_reload=True,representative_uploaded_image_preprocessing=True,test_n=len(parts['test']),
        no_test_augmentation=True,zero_division=0))
    print(json.dumps({k:v['test']['accuracy'] for k,v in summary['models'].items()}),flush=True)

if __name__=='__main__': main()
