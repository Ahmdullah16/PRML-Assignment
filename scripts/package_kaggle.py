"""Build evidence-derived PDF/Markdown and an independently usable submission ZIP."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np

TITLE='Classification of Facial Expressions into Happiness, Sadness, Fear and Neutral Using CNN and GAN: A Comparative Study'
ROLL='264102307'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=Path('results/runs/kaggle_seed42'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=Path.cwd(); out=root/'results/kaggle_seed42'
    s=json.loads((out/'summary.json').read_text()); d=json.loads((out/'dataset.json').read_text())
    verification=json.loads((out/'verification.json').read_text())
    assert s['status']=='complete' and verification['saved_predictions_recomputed']
    m=s['models']; a.output.mkdir(parents=True,exist_ok=True)
    folder=a.output/f'Ahmdullah_{ROLL}'
    folder.mkdir(exist_ok=True)
    for name in ('Code','Confusion_Matrices','Results','Graphs','Documentation','Checkpoints'):
        (folder/name).mkdir(exist_ok=True)
    dataset_license=Path(s['configuration']['config']['data'])/'LICENSE'
    if dataset_license.is_file():
        shutil.copy2(dataset_license,folder/'Documentation/Dataset_LICENSE.txt')
    for f in out.glob('*.csv'): shutil.copy2(f,folder/'Results'/f.name)
    for f in out.glob('*.json'): shutil.copy2(f,folder/'Results'/f.name)
    for f in (out/'figures').glob('*.png'):
        shutil.copy2(f,folder/('Confusion_Matrices' if 'Confusion' in f.name else 'Graphs')/f.name)
    # Include complete package modules and executable entry points, never private photographs.
    for package in ('prml','scripts','app'):
        for f in (root/package).rglob('*.py'):
            target=folder/'Code'/f.relative_to(root)
            target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,target)
    aliases={'CNN_Model.py':'from prml.models import CNN\n','GAN_Model.py':'from prml.models import Generator, Discriminator\n',
        'Train_CNN.py':"import sys\nfrom scripts.train_kaggle import main\nif __name__ == '__main__':\n    sys.argv.extend(['--model','cnn'])\n    main()\n",
        'Train_GAN.py':"import sys\nfrom scripts.train_kaggle import main\nif __name__ == '__main__':\n    sys.argv.extend(['--model','gan'])\n    main()\n",
        'Evaluate_Models.py':"from scripts.evaluate_kaggle import main\nif __name__ == '__main__': main()\n",
        'Predict.py':"from prml.predict import main\nif __name__ == '__main__': main()\n",
        'Data_Preprocessing.py':"from scripts.prepare_kaggle import prepare\nif __name__ == '__main__':\n    import runpy\n    runpy.run_module('scripts.prepare_kaggle',run_name='__main__')\n"}
    for name,text in aliases.items(): (folder/'Code'/name).write_text(text,encoding='utf-8')
    # The standalone dashboard reads the submission's own Results/Checkpoints.
    standalone_app='''import sys
from pathlib import Path
import streamlit as st
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from app.kaggle_dashboard import render
ROOT=CODE.parent
PAGES=['Overview','Dataset Explorer','CNN Explorer','GAN Explorer','Model Comparison','Prediction Studio','Research Report']
@st.cache_resource
def checkpoint(path,modified):
    import torch
    torch.set_num_threads(2)
    from prml.inference import load_checkpoint
    return load_checkpoint(path)
st.set_page_config(page_title='PRML | Kaggle Expression Lab',layout='wide')
page=st.sidebar.radio('Research workspace',PAGES)
render(page,ROOT,checkpoint,out=ROOT/'Results',report_path=ROOT/'RESULTS_REPORT.md',checkpoint_root=ROOT/'Checkpoints')
'''
    (folder/'Code/app/streamlit_app.py').write_text(standalone_app,encoding='utf-8')
    import torch,pandas,sklearn,matplotlib,PIL,streamlit,reportlab,pyarrow
    requirements='\n'.join(f'{name}=={version}' for name,version in
        [('torch',str(torch.__version__).split('+')[0]),('numpy',np.__version__),('pandas',pandas.__version__),
         ('scikit-learn',sklearn.__version__),('matplotlib',matplotlib.__version__),('Pillow',PIL.__version__),
         ('streamlit',streamlit.__version__),('reportlab',reportlab.Version),('PyYAML','6.0.3'),('pyarrow',pyarrow.__version__)])+'\n'
    (folder/'Code/requirements.txt').write_text(requirements,encoding='utf-8')
    for kind in ('cnn','gan','generator'): shutil.copy2(a.run/f'{kind}.pt',folder/'Checkpoints'/f'{kind}.pt')
    comparison=[['Metric','CNN','AC-GAN classifier']]
    for label,key in [('Accuracy','accuracy'),('Precision (Macro)','macro_precision'),('Recall (Macro)','macro_recall'),('F1-score (Macro)','macro_f1'),
                      ('Balanced accuracy','balanced_accuracy'),('Precision (Weighted)','weighted_precision'),('Recall (Weighted)','weighted_recall'),('F1-score (Weighted)','weighted_f1')]:
        comparison.append([label]+[f"{m[k]['test'][key]:.6f}" for k in ('cnn','gan')])
    perclass={kind:[['Expression','Precision','Recall','F1','Support']]+[
        [c.title()]+[f"{m[kind]['classification_report'][c][key]:.4f}" for key in ('precision','recall','f1-score')]+
        [str(int(m[kind]['classification_report'][c]['support']))] for c in ('happiness','sadness','fear','neutral')]
        for kind in ('cnn','gan')}
    counts=[['Expression','Training','Validation','Test']]+[[c.title()]+[str(d['counts'][part][c]) for part in ('train','validation','test')]
        for c in ('happiness','sadness','fear','neutral')]+[['Total']+[str(d['totals'][p]) for p in ('train','validation','test')]]
    training=[['Setting','CNN','AC-GAN']]
    for label,key in [('Executed epochs','epochs'),('Selected epoch','best_epoch'),('Classifier parameters','parameters'),('Generator parameters','generator_parameters')]:
        training.append([label]+[str(m[k][key]) for k in ('cnn','gan')])
    training.extend([['Duration (seconds)']+[f"{m[k]['training_seconds']:.2f}" for k in ('cnn','gan')],
        ['Maximum epochs','20','15'],['Batch size','64','64'],['Initial learning rate','0.001','0.0002'],['Early stopping patience','5','5']])
    winner='CNN' if m['cnn']['test']['macro_f1']>m['gan']['test']['macro_f1'] else 'AC-GAN classifier'
    gap=m['cnn']['test']['accuracy']-m['gan']['test']['accuracy']
    repairs=json.loads((out/'repair_provenance.json').read_text()) if (out/'repair_provenance.json').exists() else {'repairs':[]}
    sections=[
      ('1. Assignment objective',
       'Retrain a supervised convolutional neural network and an Auxiliary Classifier GAN from scratch to classify images into Happiness, Sadness, Fear and Neutral. Compare the CNN output with the discriminator auxiliary class head on identical held-out images. The generator is not a classifier.'),
      ('2. Dataset information and provenance',
       f"Source: dilkushsingh/facial-emotion-dataset on Kaggle, with documented upstream dilkushsingh/Facial_Emotion_Dataset. Local download contains train_dir and test_dir. The selected original folders contain {sum(v for k,v in d['source_counts'].items() if k.split('/')[-1] in ('happy','sad','fear','neutral')):,} files. All valid selected images are 48x48 JPEGs. The local LICENSE is CC BY 4.0. The Kaggle API listed version 17, last updated 2026-10-17; the local download's exact Kaggle version cannot be proven. SHA256 manifests identify the actual experimental snapshot.\n\n"
       f"The download contained {len(repairs['repairs'])} invalid training JPEGs whose contents were GitHub HTTP 429 error text. Valid files were recovered from the documented upstream archive into a private working copy, leaving the original unchanged. Repaired hashes and archive hash are recorded in repair_provenance.json. All 25,053 selected file hashes match the documented upstream archive byte-for-byte; observed upstream commit 444c3b5871d5404012abec91466ce1817df66af9 is recorded in upstream_comparison.json. No held-out file needed repair. Raw facial photographs are excluded from the repository and submission."),
      ('3. Four selected classes and label mapping',
       'Folders happy, sad, fear and neutral map to Happiness, Sadness, Fear and Neutral. Angry, disgust and surprise are excluded. For compatibility with the existing checkpoints, numeric labels are sadness=0, happiness=1, neutral=2, fear=3. Confusion-matrix figures reorder both axes to Happiness, Sadness, Fear, Neutral; saved metric matrices explicitly retain numeric label order.'),
      ('4. Training, validation and test protocol',
       f"Preserve every valid selected original test image: {d['totals']['test']:,} test records. Validation is approximately 15% of eligible original training data, stratified by class with seed 42 and entire duplicate groups assigned together. Exact duplicates use decoded RGB SHA256; near-duplicate candidates use {d['near_duplicate_rule']}. "
       f"The audit found {len(d['exact_duplicate_groups'])} exact-duplicate groups and {len(d['near_duplicate_pairs'])} conservative near-duplicate pairs. Training groups touching test are excluded, redundant exact training copies removed, and conflicting-label training groups excluded: {len(d['excluded_training'])} files in total. "
       'No file, exact pixel hash, or audited duplicate group overlaps partitions. The 5,009 original test records contain 4,971 unique decoded pixel hashes (38 repeated records); one exact-pixel test group has conflicting source labels. Original test membership and annotations are retained for both classifiers. These issues reduce effective sample independence and expose annotation uncertainty; details are in test_duplicate_summary.json. Subject identifiers are unavailable. These are image-level results, not subject-independent estimates. Heuristics cannot guarantee absence of every near duplicate. Both models use exactly the same saved manifest; test labels do not select hyperparameters or checkpoints.'),
      ('5. Image preprocessing',
       'Apply EXIF orientation correction, RGB conversion, bilinear resize to 64x64 and float32 scaling x/127.5 - 1. Both models use random horizontal flips only during training. Validation and test preprocessing are deterministic with no augmentation. A uint8 training/validation cache avoids repeated decoding. No test pixels are loaded during the training program.'),
      ('6. CNN architecture and mathematical formulation',
       'The existing CNN is reused unchanged: three convolution 3x3 / batch normalization / ReLU / max-pool blocks with 16,32,64 channels. At 64x64 input, flatten the 64x8x8 features to 4096, then dense 64, ReLU, dropout 0.3, and dense 4 logits.\n\n'
       'p(y=k|x) = exp(z_k) / sum_j exp(z_j). L_CNN = -log p(y|x). Training uses cross-entropy on logits; softmax is applied at inference. Scores are uncalibrated.'),
      ('7. AC-GAN architecture and mathematical formulation',
       'The existing conditional generator concatenates 64-dimensional Gaussian noise and a 16-dimensional class embedding, projects to 256x4x4, then applies four transposed-convolution stages with channels 128,64,32,3. Batch normalization and ReLU precede the final Tanh RGB 64x64 output. The discriminator has four stride-2 4x4 convolutions with 32,64,128,256 channels and LeakyReLU 0.2. Flatten 4096 features into a source logit and four expression logits.\n\n'
       'L_D = BCE(D_s(x),1) + BCE(D_s(G(z,c)),0) + CE(D_c(x),y) + CE(D_c(G(z,c)),c). '
       'L_G = BCE(D_s(G(z,c)),1) + CE(D_c(G(z,c)),c). All loss weights are one. BCE uses logits and the generator objective is non-saturating. Fake images are detached during the D update; D parameters are frozen during the G update while preserving derivatives through D to G. Both gradients are checked for finiteness, generator gradients for nonzero propagation, and gradient norms clipped to 10. Generator and discriminator snapshots use the same validation-selected epoch. Generated grids remain private because image quality and possible training-image memorization are not established.'),
      ('8. Training configuration and hardware',
       'Both methods start from random weights with seed 42 on the full eligible training partition. CNN uses AdamW, lr 0.001, betas (0.9,0.999), weight decay 0.0001; ReduceLROnPlateau monitors validation CE, factor 0.5, patience 2, floor 1e-5. GAN D uses AdamW with zero weight decay and G uses Adam; both lr 0.0002 and betas (0.5,0.999), fixed LR. Batch size 64. Budget: 20 CNN epochs, 15 GAN epochs; early stopping after five validation epochs without a CE improvement of at least 1e-5. Minimum validation CE selects checkpoints. This is a single-seed, fixed-budget classroom comparison, without architecture-matched ablation or exhaustive tuning.\n\n'
       'Hardware: Intel Core i5-1135G7, 4 physical cores / 8 logical processors; CPU execution with four PyTorch threads; Intel Iris Xe integrated graphics, no CUDA GPU. Environment: '+json.dumps(s['configuration']['environment'])+'. Epoch duration includes in-epoch validation; total training duration excludes initial image caching and final test evaluation.'),
      ('9. Evaluation metrics and arithmetic validation',
       'Accuracy = number of correct predictions / N. For class k: precision P_k = TP_k/(TP_k+FP_k), recall R_k = TP_k/(TP_k+FN_k), F1_k = 2*P_k*R_k/(P_k+R_k). Undefined divisions are assigned zero. Macro metrics are the arithmetic mean across all four classes: F1_macro = (F1_1 + F1_2 + F1_3 + F1_4)/4. Weighted F1 = sum_k (n_k/N)*F1_k, where n_k is test support. Balanced accuracy is macro recall. Reported metrics are independently cross-checked with scikit-learn and recomputed from saved test predictions. Confusion matrices count all eligible test records. The source-order matrices and labels are in summary.json.'),
      ('10. CNN confusion matrix',
       'Rows are actual expressions; columns are predicted expressions. Raw counts are shown, with a separate row-normalized matrix supplied. The matrix derives from the validation-selected CNN checkpoint and saved held-out predictions.'),
      ('11. GAN classification confusion matrix',
       'The matrix uses predictions from the discriminator auxiliary class head, not generated labels. Its rows and columns follow the same order and use the same held-out image list as the CNN.'),
      ('12. Accuracy, precision, recall and F1 comparison',
       f"The measured comparison is shown below. Accuracy difference (CNN minus GAN) is {gap*100:.2f} percentage points. Precision, recall and F1 in the main comparison are macro averages; weighted results are also supplied. Each model has {d['totals']['test']:,} test predictions."),
      ('13. Training graphs and interpretation',
       'Training accuracy is aggregated from augmented minibatches while weights change; validation accuracy is measured in eval mode after each epoch. CNN train and validation losses are classification CE. GAN discriminator total loss contains four terms and generator total loss contains two; their magnitudes cannot be interpreted as equivalent to CNN CE. The selected epoch is indicated on accuracy plots. GAN losses and numerical gradient checks do not establish visual quality, sample diversity or freedom from mode collapse.'),
      ('14. Results discussion and limitations',
       f"{winner} obtains the higher test macro F1 in this executed run. A constant predictor choosing the most frequent training class ({s['training_majority_baseline']['selected_class']}) scores {s['training_majority_baseline']['accuracy']:.2%} test accuracy and {s['training_majority_baseline']['macro_f1']:.4f} macro F1; this baseline class is chosen from training counts only. The measured differences are descriptive evidence for these particular architectures, budgets, split and seed. They do not establish that one learning paradigm is universally superior. The CNN directly optimizes real-image recognition; AC-GAN shares representation capacity across real/fake discrimination and expression classification. Those objectives may compete. Architecture, parameter counts and adversarial compute differ, so this is a pipeline comparison rather than an isolated causal test of adversarial learning. Class imbalance makes macro and per-class metrics especially relevant.\n\n"
       'Unavailable identities, retained within-test duplicates, imperfect duplicate detection, dataset annotation noise and a single random seed limit generalization claims. The supplied test set was used only after both training procedures completed. A private fixed-seed generator grid shows coarse face-like images and low within-class variation, particularly for sadness and happiness, consistent with partial mode collapse. This qualitative diagnostic does not establish diversity or absence of memorization. No quantitative synthesis-quality, calibration, demographic-fairness or real-world-robustness assessment was performed. These limitations do not invalidate the measured discriminator classification predictions, but prevent claims of successful high-quality facial synthesis. This report does not claim an accuracy improvement over previous 48-image studies, because their datasets and protocols differ.'),
      ('15. Conclusion',
       'The assignment implements genuine supervised CNN training and conditional adversarial G/D training, saves validation-selected consistent checkpoints, and compares both classifiers on the same held-out Kaggle partition. Reports, metrics, histories, exact splits and executable source are packaged to support reproduction. Predictions label facial appearance and do not establish actual emotion or mental state.'),
      ('16. Attribution and references',
       'Dilkush Singh. Facial Emotion Dataset. Kaggle: https://www.kaggle.com/datasets/dilkushsingh/facial-emotion-dataset. Documented upstream: https://github.com/dilkushsingh/Facial_Emotion_Dataset. Local metadata license: Creative Commons Attribution 4.0; raw source photographs are not redistributed.\n\n'
       'LeCun et al. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE. DOI: 10.1109/5.726791.\n\n'
       'Goodfellow et al. (2014). Generative Adversarial Nets. NeurIPS. https://arxiv.org/abs/1406.2661.\n\n'
       'Odena, Olah and Shlens (2017). Conditional Image Synthesis with Auxiliary Classifier GANs. ICML / PMLR 70:2642-2651. https://proceedings.mlr.press/v70/odena17a.html.\n\n'
       'Paszke et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS. https://arxiv.org/abs/1912.01703.')]
    def mdtable(rows): return '\n'.join(['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(r)+' |' for r in rows[1:]])
    image_sections={10:['CNN_Confusion_Matrix.png'],11:['GAN_Confusion_Matrix.png'],13:['CNN_Training_Accuracy.png','CNN_Training_Loss.png','GAN_Training_Loss.png','GAN_Classification_Accuracy.png','Accuracy_Comparison.png','Precision_Comparison.png','Recall_Comparison.png','F1_Score_Comparison.png']}
    md=f'# {TITLE}\n\n**Student:** Ahmdullah\n\n**Roll Number:** {ROLL}\n\n**Institution:** IIT Guwahati\n\n**Course:** Pattern Recognition and Machine Learning\n\n**Program:** M.Tech - Signal Processing and Machine Learning\n\n**Department:** Electronics and Electrical Engineering\n\n'
    for i,(heading,text) in enumerate(sections,1):
        md+=f'## {heading}\n\n{text}\n\n'
        if i==4: md+=mdtable(counts)+'\n\n'
        if i==8: md+=mdtable(training)+'\n\n'
        if i==12:
            md+=mdtable(comparison)+'\n\n'
            for kind,rows in perclass.items(): md+=f'### {kind.upper()} per-class test metrics\n\n'+mdtable(rows)+'\n\n'
        for file in image_sections.get(i,[]):
            sub='Confusion_Matrices' if 'Confusion' in file else 'Graphs'
            md+=f'![{file.replace("_"," ")}]({sub}/{file})\n\n'
    (folder/'RESULTS_REPORT.md').write_text(md,encoding='utf-8')
    # Deliberately avoid non-ASCII math glyphs in the PDF's standard embedded fonts.
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Topic',fontName='Helvetica-Bold',fontSize=20,leading=25,textColor=colors.HexColor('#16384c'),spaceAfter=22))
    styles['BodyText'].fontSize=10; styles['BodyText'].leading=15; styles['BodyText'].spaceAfter=10
    styles['Heading2'].fontSize=14; styles['Heading2'].leading=19; styles['Heading2'].spaceBefore=18
    story=[Paragraph(escape(TITLE),styles['Topic']),Paragraph('Student: Ahmdullah | Roll Number: '+ROLL,styles['Heading2']),
        Paragraph('Institution: IIT Guwahati<br/>Course: Pattern Recognition and Machine Learning<br/>Program: M.Tech - Signal Processing and Machine Learning<br/>Department: Electronics and Electrical Engineering',styles['BodyText']),Spacer(1,14),
        Paragraph('NEW KAGGLE EXPERIMENT - MEASURED HELD-OUT RESULTS',styles['Heading3'])]
    def table(rows):
        width=487/len(rows[0])
        cells=[[Paragraph(escape(c),styles['BodyText']) for c in row] for row in rows]
        t=Table(cells,colWidths=[width]*len(rows[0]),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3edf2')),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#167e87')),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f8fa')])]))
        return t
    story.append(table(comparison[:5])); story.append(Spacer(1,18))
    story.append(Paragraph(f"Same {d['totals']['test']:,} held-out test images for both classifiers. Validation-only checkpoint selection. Historical 48-image studies remain separate.",styles['BodyText']))
    story.append(PageBreak())
    for i,(heading,text) in enumerate(sections,1):
        if i in (6,8,10,11,12,13,14,16): story.append(PageBreak())
        story.append(Paragraph(escape(heading),styles['Heading2']))
        for para in text.split('\n\n'): story.append(Paragraph(escape(para),styles['BodyText']))
        if i==4: story.append(KeepTogether([table(counts)]))
        if i==8: story.append(table(training))
        if i==12:
            story.append(table(comparison))
            for kind,rows in perclass.items():
                story.append(Paragraph(kind.upper()+' per-class test metrics',styles['Heading3']))
                story.append(table(rows))
        for file in image_sections.get(i,[]):
            img=Image(str(out/'figures'/file)); ratio=img.imageHeight/img.imageWidth
            image=Image(str(out/'figures'/file),width=450,height=450*ratio)
            story.append(KeepTogether([Spacer(1,12),image,Paragraph(escape(file.replace('_',' ').removesuffix('.png')),styles['Caption'] if 'Caption' in styles else styles['BodyText'])]))
    def footer(canvas,doc):
        canvas.saveState(); canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#516776'))
        canvas.drawString(54,30,'Ahmdullah | 264102307 | PRML | Kaggle four-class experiment')
        canvas.drawRightString(A4[0]-54,30,f'Page {doc.page}'); canvas.restoreState()
    SimpleDocTemplate(str(folder/'RESULTS_REPORT.pdf'),pagesize=A4,rightMargin=54,leftMargin=54,topMargin=46,bottomMargin=49,
        title=TITLE,author='Ahmdullah').build(story,onFirstPage=footer,onLaterPages=footer)
    reproduction='''# Reproduction instructions

Python 3.12 is required. Create a virtual environment with `python -m venv .venv`, then use `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Linux/macOS) in place of `python` below. From the Code directory:

```bash
python -m pip install -r requirements.txt
python -m scripts.prepare_kaggle --data "PATH/TO/DATA SET" --output ../Reproduced_Results
python -m scripts.train_kaggle --data "PATH/TO/DATA SET" --manifest ../Reproduced_Results/split_manifest.json --output ../Reproduced_Training_Run
python -m scripts.evaluate_kaggle --data "PATH/TO/DATA SET" --run ../Reproduced_Training_Run --output ../Reproduced_Results
python -m prml.predict --checkpoint ../Checkpoints/cnn.pt --image "PATH/TO/IMAGE.jpg"
python -m prml.predict --checkpoint ../Checkpoints/gan.pt --image "PATH/TO/IMAGE.jpg"
python -m scripts.verify_submission --root ..
```

Use the included exact manifest to reproduce the measured snapshot; compare the regenerated manifest hash against Results/split_manifest.json before training. Preparation used the versions in audit_environment.json; training/evaluation versions are recorded separately. Regenerating manifests is appropriate only for the same hash-verified dataset or a separately named new experiment. The original Kaggle download had 393 invalid training JPEGs (HTTP 429 text). To recover the measured snapshot, download the documented upstream ZIP, use scripts.repair_kaggle with the included original_download_audit.json, then rerun preparation against the private repaired copy. repair_provenance.json specifies repaired file hashes and archive hash. Do not mix different input snapshots while reusing an existing training run. The reproduction commands use fresh output directories and preserve the included measured Results directory.

Train_CNN.py and Train_GAN.py are convenience launchers; all imported project modules are included under Code. CNN_Model.py and GAN_Model.py expose the implementations in prml/models.py. Evaluation compares all eligible supplied test images. The included manifest records test membership. Training resume states are private and excluded from submission; the selected CNN, discriminator and paired generator checkpoints are included. No training is needed to replay inference.

Launch the standalone seven-section dashboard from Code with `python -m streamlit run app/streamlit_app.py`. Its entry point reads this submission's Results and Checkpoints folders. The GitHub dashboard additionally preserves the historical experiment selector. Training, inference, evaluation and the submission dashboard are independently usable without a repository checkout.
'''
    (folder/'Documentation/Reproduction_Instructions.md').write_text(reproduction,encoding='utf-8')
    (folder/'Documentation/Dataset_Information.md').write_text('\n\n'.join(text for _,text in sections[1:5]),encoding='utf-8')
    (folder/'Documentation/Training_Configuration.md').write_text(sections[7][1]+'\n\n'+mdtable(training),encoding='utf-8')
    readme=f'''# PRML assignment - Ahmdullah ({ROLL})

{TITLE}

Start with RESULTS_REPORT.pdf or RESULTS_REPORT.md. All results come from the newly executed Kaggle four-class experiment, not the earlier 48-image baseline. Code contains complete importable modules and runnable launchers. Results includes exact split manifests, test probabilities, independently verified metrics, histories, audit and provenance. Checkpoints includes the validation-selected CNN, AC-GAN discriminator and paired generator. Raw facial photographs are excluded.

See Documentation/Reproduction_Instructions.md for setup and commands. Run `python -m scripts.verify_submission --root ..` from Code to check hashes, split separation, saved metrics, probabilities and paired checkpoint loading without source photographs. Precision, recall and F1 in the main table are macro averages; undefined divisions use zero. Subject independence is not established. Predicted expressions do not establish actual emotional state.

Repository: https://github.com/Ahmdullah16/PRML-Assignment
Dashboard target: https://facial-expression.streamlit.app/
'''
    (folder/'README.md').write_text(readme,encoding='utf-8')
    reportdir=root/'reports/kaggle_seed42'; reportdir.mkdir(parents=True,exist_ok=True)
    # Repository report uses paths relative to its own directory.
    repo_md=md.replace('(Confusion_Matrices/','(../../results/kaggle_seed42/figures/').replace('(Graphs/','(../../results/kaggle_seed42/figures/')
    (reportdir/'RESULTS_REPORT.md').write_text(repo_md,encoding='utf-8')
    shutil.copy2(folder/'RESULTS_REPORT.pdf',reportdir/'RESULTS_REPORT.pdf')
    shutil.copy2(folder/'RESULTS_REPORT.pdf',a.output/'RESULTS_REPORT.pdf')
    shutil.copy2(folder/'RESULTS_REPORT.md',a.output/'RESULTS_REPORT.md')
    for k in ('CNN','GAN'): shutil.copy2(folder/f'Confusion_Matrices/{k}_Confusion_Matrix.png',a.output/f'{k}_Confusion_Matrix.png')
    required=['RESULTS_REPORT.pdf','RESULTS_REPORT.md','Code/prml/models.py','Code/prml/data.py','Code/scripts/train_kaggle.py',
        'Confusion_Matrices/CNN_Confusion_Matrix.png','Confusion_Matrices/GAN_Confusion_Matrix.png','Results/CNN_Metrics.csv',
        'Results/GAN_Metrics.csv','Results/Model_Comparison.csv','Results/Test_Predictions.csv','Results/Experiment_Configuration.json']
    assert all((folder/f).is_file() for f in required)
    hashes={p.relative_to(folder).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*')
            if p.is_file() and p.relative_to(folder).as_posix()!='Documentation/file_sha256.json'}
    (folder/'Documentation/file_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    archive=a.output/f'Ahmdullah_{ROLL}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in folder.rglob('*'):
            if f.is_file(): z.write(f,f.relative_to(a.output))
    extracted=a.output.parent/'work/zip_validation'
    extracted.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        z.extractall(extracted)
    for name,digest in hashes.items():
        assert hashlib.sha256((extracted/folder.name/name).read_bytes()).hexdigest()==digest
    print(json.dumps(dict(archive=str(archive),files=len(hashes),bytes=archive.stat().st_size,zip_integrity=True)),flush=True)

if __name__=='__main__': main()
