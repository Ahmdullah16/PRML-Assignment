"""Seven-section dashboard for the separately measured Kaggle experiment."""
import io
import hashlib
import json
import os
import re
from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st

def render(page,root,checkpoint,out=None,report_path=None,checkpoint_root=None):
    out=out if out is not None else root/'results/kaggle_seed42'
    summary=json.loads((out/'summary.json').read_text())
    dataset=json.loads((out/'dataset.json').read_text())
    models=summary['models']
    st.caption('KAGGLE FOUR-CLASS RETRAINING · SEED 42 · IMAGE-LEVEL EVALUATION')
    st.title(page)
    st.caption('Ahmdullah · Roll 264102307 · IIT Guwahati · PRML')
    def figure(name,caption):
        p=out/'figures'/name
        if not p.exists():
            p=root/('Confusion_Matrices' if 'Confusion' in name else 'Graphs')/name
        if p.exists(): st.image(str(p),caption=caption,width='stretch')
    comparison=pd.DataFrame({k.upper():{m:v['test'][m] for m in
        ('accuracy','macro_precision','macro_recall','macro_f1','balanced_accuracy','weighted_f1')} for k,v in models.items()})
    if page=='Overview':
        st.subheader('CNN and AC-GAN classification on the same held-out images')
        st.write('Both models were retrained from scratch on the four selected classes of dilkushsingh/facial-emotion-dataset. The GAN classifier is its discriminator auxiliary head.')
        cols=st.columns(3)
        for col,(name,n) in zip(cols,dataset['totals'].items()): col.metric(name.title()+' images',n)
        for col,k in zip(st.columns(2),('cnn','gan')): col.metric(k.upper()+' test accuracy',f"{models[k]['test']['accuracy']:.2%}")
        st.success('Training and held-out evaluation complete; metrics cross-checked against saved predictions.')
        st.info("These labels describe apparent facial expressions and do not establish a person's actual emotional or mental state.")
        st.write('Historical 48-image experiments remain separately accessible through the Evidence source selector.')
    elif page=='Dataset Explorer':
        st.dataframe(pd.DataFrame(dataset['counts']).fillna(0).astype(int),width='stretch')
        st.bar_chart(pd.DataFrame(dataset['counts']))
        st.write(dataset['preprocessing'])
        st.write('Provided test membership is preserved. Validation uses approximately 15% of eligible source training images, stratified by class and grouped by duplicates. Training groups that overlap test groups are excluded.')
        st.write(f"Exact duplicate groups: {len(dataset['exact_duplicate_groups'])}; conservative near-duplicate pairs: {len(dataset['near_duplicate_pairs'])}; excluded training files: {len(dataset['excluded_training'])}.")
        st.caption(dataset['near_duplicate_rule'])
        st.warning(dataset['limitations'])
        st.markdown('[Kaggle dataset](https://www.kaggle.com/datasets/dilkushsingh/facial-emotion-dataset) · [Documented upstream](https://github.com/dilkushsingh/Facial_Emotion_Dataset)')
        provenance=out/'repair_provenance.json'
        if provenance.exists():
            repairs=json.loads(provenance.read_text())['repairs']
            st.write(f'{len(repairs)} invalid JPEG error-text payloads in the download were recovered from the documented upstream in a private working copy. Original download preserved; repairs are hash-recorded.')
        st.caption(dataset['version'])
    elif page in ('CNN Explorer','GAN Explorer'):
        kind='cnn' if page=='CNN Explorer' else 'gan'; name=kind.upper()
        st.dataframe(pd.DataFrame({'Value':models[kind]['test']}).loc[['accuracy','macro_precision','macro_recall','macro_f1','balanced_accuracy']])
        st.json({k:models[kind][k] for k in ('epochs','best_epoch','training_seconds','parameters','generator_parameters')})
        if kind=='cnn':
            st.write('RGB 64×64 → Conv/BN/ReLU/Pool blocks (16,32,64) → flatten 4096 → dense 64/ReLU/dropout 0.3 → four logits. Cross-entropy, AdamW, validation-loss learning-rate reduction and early stopping.')
            st.latex(r'p(y=k|x)=\mathrm{softmax}(f_\theta(x))_k,\quad L_{CNN}=-\log p(y|x)')
            figure('CNN_Training_Accuracy.png','Augmented training minibatch accuracy and unaugmented validation accuracy')
        else:
            st.write('Generator: 64 noise values plus a 16-dimensional class embedding → dense 256×4×4 → transposed convolutions 128/64/32/3 → Tanh. Discriminator: stride-2 convolution channels 32/64/128/256, LeakyReLU, independent source and four-class heads.')
            st.latex(r'L_D=BCE(D_s(x),1)+BCE(D_s(G(z,c)),0)+CE(D_c(x),y)+CE(D_c(G(z,c)),c)')
            st.latex(r'L_G=BCE(D_s(G(z,c)),1)+CE(D_c(G(z,c)),c)')
            st.write('Generator/discriminator snapshots share the validation-selected epoch. Gradients and losses are checked for finiteness; gradient norm clipping is 10. Synthetic samples remain private; visual quality and memorization are not established.')
            st.caption('Private sample inspection found coarse face-like outputs and limited within-class variation, consistent with partial mode collapse. No quantitative generation-quality claim is made.')
            figure('GAN_Classification_Accuracy.png','Discriminator auxiliary classifier accuracy')
        figure(name+'_Training_Loss.png','Objectives have different definitions across models')
        figure(name+'_Confusion_Matrix.png','Actual rows; predicted columns; all eligible test images')
        st.json(summary['configuration']['config'])
    elif page=='Model Comparison':
        st.dataframe(comparison.style.format('{:.4f}'),width='stretch')
        st.caption('Precision, recall and F1 are macro averages. All models use the same splits; zero division is assigned zero.')
        st.dataframe(pd.DataFrame({k.upper():{m:v[m] for m in ('epochs','best_epoch','training_seconds','parameters')} for k,v in models.items()}))
        for col,k in zip(st.columns(2),('CNN','GAN')):
            with col: figure(k+'_Confusion_Matrix.png',k+' held-out counts')
        for f in ('Accuracy','Precision','Recall','F1_Score'): figure(f+'_Comparison.png','Same test images and metric definitions')
        st.download_button('Download measured comparison',(out/'Model_Comparison.csv').read_bytes(),'Model_Comparison.csv','text/csv')
    elif page=='Prediction Studio':
        st.write('Upload a single cropped face image. The models resize the full image and do not detect faces.')
        st.info("Apparent expression categories do not establish actual emotional state. Scores are uncalibrated softmax outputs. Uploads are processed in memory.")
        choice=st.selectbox('Model',['Both models','CNN','AC-GAN classifier'])
        kinds=('cnn','gan') if choice=='Both models' else (('cnn',) if choice=='CNN' else ('gan',))
        directory=Path(os.getenv('PRML_CHECKPOINT_DIR',str(checkpoint_root if checkpoint_root is not None else root/'deployment')))
        paths={k:directory/f'{k}.pt' for k in kinds}
        ready=all(p.is_file() for p in paths.values())
        if not ready: st.warning('Prediction unavailable: trained checkpoints have not been provisioned on this server.')
        uploaded=st.file_uploader('Image (PNG or JPEG)',type=['png','jpg','jpeg'],disabled=not ready)
        if uploaded is not None and ready:
            try:
                from prml.inference import predict
                raw=uploaded.getvalue()
                with Image.open(io.BytesIO(raw)) as im: im.verify()
                st.image(raw,width=220,caption='Uploaded image')
                for col,kind in zip(st.columns(len(kinds)),kinds):
                    with col:
                        actual_hash=hashlib.sha256(paths[kind].read_bytes()).hexdigest()
                        if actual_hash!=models[kind]['checkpoint_sha256']:
                            raise ValueError('Checkpoint hash does not match the verified experiment')
                        net,meta=checkpoint(str(paths[kind]),paths[kind].stat().st_mtime_ns)
                        if meta.get('experiment')!=summary['experiment']: raise ValueError('Checkpoint does not belong to the Kaggle experiment')
                        if meta.get('best_epoch')!=models[kind]['best_epoch']: raise ValueError('Checkpoint selected epoch does not match the report')
                        scores=predict(net,kind,io.BytesIO(raw))
                        st.subheader(kind.upper()); st.metric('Predicted expression',max(scores,key=scores.get).title())
                        st.bar_chart(pd.DataFrame({'Softmax score':scores}))
                        st.caption(f"{meta['experiment']} · selected epoch {meta['best_epoch']}")
            except (ValueError,OSError,RuntimeError,Image.DecompressionBombError) as exc:
                st.error(f'Unable to process image or checkpoint: {exc}')
        st.caption('RGB 64×64 bilinear resize and normalization to [-1,1], identical to training. Four labels: sadness, happiness, neutral, fear.')
    elif page=='Research Report':
        report=report_path if report_path is not None else root/'reports/kaggle_seed42/RESULTS_REPORT.md'
        if report.exists():
            st.download_button('Download Markdown report',report.read_bytes(),report.name,'text/markdown')
            for chunk in re.split(r'(!\[[^\]]*\]\([^)]*\))',report.read_text(encoding='utf-8')):
                image=re.fullmatch(r'!\[([^\]]*)\]\(([^)]*)\)',chunk)
                if image:
                    figure(Path(image[2]).name,image[1])
                elif chunk.strip():
                    st.markdown(chunk)
            pdf=report.with_suffix('.pdf')
            if pdf.exists(): st.download_button('Download PDF report',pdf.read_bytes(),pdf.name,'application/pdf')
        else: st.warning('Final report unavailable.')
