# Zeoformer

We have provided the CSV file of OSDB and the CIF files used in our experiment.
You can train and test the model with the following commands:

```bash
conda create --name zeoformer python=3.10
conda activate zeoformer
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
pip install jarvis-tools==2022.9.16
python setup.py install
# Training Zeoformer for OSDB
cd matformer/scripts/osdb
python train.py
```

## Acknowledgement
Our code is built upon previous code versions of the [Matformer](https://github.com/YKQ98/Matformer?tab=readme-ov-file) and [ALIGNN](https://github.com/YKQ98/Matformer?tab=readme-ov-file)(https://github.com/usnistgov/alignn) frameworks. We are deeply grateful for these two excellent codebases.
