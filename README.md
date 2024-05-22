# Zeoformer


You can train and test the model with the following commands:

```bash
conda create --name zeoformer python=3.10
conda activate zeoformer
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
pip install jarvis-tools==2022.9.16
python setup.py
# Training Matformer for the Materials Project
cd matformer/scripts/mp
python train.py
# Training Matformer for JARVIS
cd matformer/scripts/jarvis
python train.py
```
