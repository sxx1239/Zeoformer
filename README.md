# Zeoformer

We have provided the CSV file of OSDB and the CIF files used in our experiment.
You can train and test the model with the following commands:

```bash
conda create --name zeoformer python=3.10
conda activate zeoformer
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
pip install jarvis-tools==2022.9.16
python setup.py
# Training Zeoormer for the Materials Project
cd matformer/scripts/mp
python train.py
# Training Zeoformer for JARVIS
cd matformer/scripts/jarvis
python train.py
# Training Zeoformer for OSDB
cd matformer/scripts/jarvis
python train.py
```

## Acknowledgement
Our code is based on the previous Matformer and ALIGNN frameworks. We are very grateful for the excellent code libraries they have provided. We have not yet thoroughly organized our code, so it may appear somewhat disorganized.
