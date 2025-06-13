from functools import partial

# from pathlib import Path
from typing import Any, Dict, Union

import ignite
import torch
import time
import warnings
from ignite.contrib.handlers import TensorboardLogger

try:
    from ignite.contrib.handlers.stores import EpochOutputStore
except Exception as exp:
    from ignite.handlers.stores import EpochOutputStore

    pass
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
)
import pickle as pk
import numpy as np
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn
from zeoformer import models
from zeoformer.data import get_train_val_loaders
from zeoformer.config import TrainingConfig
from zeoformer.models.pyg_att import Zeoformer
import csv
from jarvis.db.jsonutils import dumpjson
import json
import pprint
import pandas as pd
import os
from zeoformer.graphs import PygGraph, PygStructureDataset
from torch.utils.data import DataLoader
from pymatgen.core.structure import Structure
from torch_geometric.data import Data

import pickle

config = {
    "dataset": "dft_3d",
    "target": "Binding_OSDA",
    "epochs": 300,  # 00,#00,
    "batch_size": 1,  # 0,
    "weight_decay": 1e-05,
    "learning_rate": 1e-03,
    "criterion": "l1",
    "optimizer": "adamw",
    "scheduler": "onecycle",
    "save_dataloader": False,
    "pin_memory": False,
    "write_predictions": False,
    "num_workers": 10,
    "random_seed": 123,
    "output_dir": os.path.abspath("."),
    "classification_threshold": None,
    "atom_features": "cgcnn",
    "distributed": False,
    "model": {
        "name": "zeoformer",
    },
}


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer


def train_zeo(
        config: Union[TrainingConfig, Dict[str, Any]],
        model: nn.Module = None,
        test_loader=[],
        test_only=False,
        prepare_batch=None,
):
    """
    `config` should conform to zeoformer.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    print(config)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)
            print('error in converting to training config!')
    import os

    deterministic = False
    classification = False
    print("config:")
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    pprint.pprint(tmp)
    if config.classification_threshold is not None:
        classification = True
    if config.random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(config.random_seed)

    line_graph = True

    if classification:
        config.model.classification = True
    # define network, optimizer, scheduler
    _model = {
        "zeoformer": Zeoformer,
    }

    if model is None:
        net = _model.get(config.model.name)(config.model)
        print("config:")
        pprint.pprint(config.model.dict())
    else:
        net = model

    net.to(device)
    if config.distributed:
        import torch.distributed as dist
        import os

        def setup(rank, world_size):
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            # initialize the process group
            dist.init_process_group("gloo", rank=rank, world_size=world_size)

        def cleanup():
            dist.destroy_process_group()

        setup(2, 2)
        net = torch.nn.parallel.DistributedDataParallel(
            net
        )
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)

    if config.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    elif config.scheduler == "onecycle":
        steps_per_epoch = len(test_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100000,
            gamma=0.96,
        )
    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
    }
    criterion = criteria[config.criterion]

    trainer = create_supervised_trainer(
        net,
        optimizer,
        criterion,
        prepare_batch=prepare_batch,
        device=device,
        deterministic=deterministic,
    )

    # std_train1 = 58.658446
    # mean_train1 = -139.596734

    #new
    # std_train1 = 60.018711
    # mean_train1 = -130.734573

    # 03
    # mean_train1 = -125.514320
    # std_train1 = 61.848293

    # 03fine_tune
    # mean_train1 = -125.575935
    # std_train1 = 61.937805

    # 04
    mean_train1 = -124.002579
    std_train1 = 61.241516

    # 05
    # mean_train1 = -120.952515
    # std_train1 = 59.570572
    if test_only:
        checkpoint_tmp = torch.load(
            'model004.pt')
        to_load = {
            "model": net,
            # "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            # "trainer": trainer,
        }
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint_tmp)
        net.eval()
        targets = []
        predictions = []
        # num_osda = []
        import time
        t1 = time.time()
        with torch.no_grad():
            for dat in test_loader:
                g, target = dat
                out_data = net(g.to(device))
                # 检查 out_data 是否包含 NaN 或 Inf
                if torch.isnan(out_data).any() or torch.isinf(out_data).any():
                    print("Warning: NaN or Inf detected in out_data, replacing with zero.")
                    out_data = torch.nan_to_num(out_data, nan=0.0, posinf=1e6, neginf=-1e6)
                out_data = out_data.cpu().numpy().tolist()
                target = target.cpu().numpy().flatten().tolist()
                if len(target) == 1:
                    target = target[0]
                targets.append(target)
                predictions.append(out_data)
        t2 = time.time()
        f.close()
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import mean_squared_error
        # num_osda = np.array(num_osda).reshape(-1, 1)
        targets = np.array(targets) * std_train1 + mean_train1
        predictions = np.array(predictions) * std_train1 + mean_train1
        # 检查并修复 NaN 或 Inf
        if np.isnan(targets).any() or np.isinf(targets).any():
            print("Warning: targets contain NaN or Inf, replacing with zero.")
            targets = np.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)

        if np.isnan(predictions).any() or np.isinf(predictions).any():
            print("Warning: predictions contain NaN or Inf, replacing with zero.")
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
        import csv
        with open('005_new_04model.csv', 'w') as f:
            f.write("target,prediction\n")
            writer = csv.writer(f)
            for target, pred in zip(targets, predictions):
                writer.writerow((target, pred))
        f.close()
        print("Test MAE:", mean_absolute_error(targets, predictions))
        print("Test RMSE:", np.sqrt(mean_squared_error(targets, predictions)))
        print("Total test time:", t2 - t1)
        return mean_absolute_error(targets, predictions)


def load_pyg_graphs(
    df: pd.DataFrame,
    name: str = "dft_3d",
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8,
    max_neighbors: int = 12,
    # cachedir: Optional[Path] = None,
    use_canonize: bool = False,
    use_lattice: bool = False,
    use_angle: bool = False,
):
    # 定义文件路径
    features_total_file = 'features_total_all12.pkl'
    # 如果文件存在，则加载 features_smiles 字典
    if os.path.exists(features_total_file):
        # with open(features_total_file, 'rb') as f:
        #     features_total = pk.load(f)
        print("zhaodao")

    else:
        features_total = {}

    # graphs = []
    def distance(pos1, pos2, lattice_matrix):
        """计算两个原子之间的距离，考虑晶格参数"""
        delta = pos2 - pos1
        delta = np.dot(delta, np.linalg.inv(lattice_matrix))
        delta -= np.round(delta)
        delta = np.dot(delta, lattice_matrix)
        dist = np.linalg.norm(delta)
        return dist

    def extract_features_cif(crystal_id, cutoff: float = 5):
        filename = f"features_{crystal_id}.pkl"

        # 检查文件是否存在
        if os.path.exists(filename):
            # 文件存在，读取文件
            with open(filename, 'rb') as f:
                g = pickle.load(f)
                features = PygStructureDataset._get_attribute_lookup("cgcnn")
                g.x, g.edge_index, g.edge_attr, g.adj = torch.tensor(g.x), torch.tensor(
                    g.edge_index), torch.tensor(g.edge_attr), torch.tensor(g.adj)
                z = g.x
                g.atomic_number = z
                z = z.type(torch.IntTensor).squeeze()
                f = torch.tensor(features[z]).type(torch.FloatTensor)
                if g.x.size(0) == 1:
                    f = f.unsqueeze(0)
                g.x = f
                # 设置边特征和邻接矩阵
                g.edge_attr = g.edge_attr.float()
                g.adj = torch.mm(g.adj, g.x) / 1000
                g.x = g.x + g.adj
                feature_cs = Data()
                feature_cs.x, feature_cs.edge_index, feature_cs.edge_attr = g.x, g.edge_index, g.edge_attr
            return feature_cs
        absolute_path = str(crystal_id)+ '.cif'

        crystal = Structure.from_file(absolute_path, primitive=False)
        atom_features = np.vstack([crystal[i].specie.number for i in range(len(crystal))])

        all_nbrs = crystal.get_all_neighbors(cutoff, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        min_nbrs = min(len(neighborlist) for neighborlist in all_nbrs)
        # if min_nbrs >= 12 & cutoff == 5:
        #     return features_total[crystal_id]
        if min_nbrs < 12:
            return extract_features_cif(
                crystal_id,
                cutoff=cutoff + 1,
            )
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:12])))
            nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:12])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = nbr_fea.astype(np.float32)
        m, n = nbr_fea_idx.shape
        # 生成对应的一维数组
        arr = np.repeat(np.arange(m), n)
        arr = np.array(arr)
        nbr_fea_idx = nbr_fea_idx.reshape(m * n)
        nbr_fea_idx = np.stack((arr, nbr_fea_idx), axis=0).astype(np.int64)
        nbr_fea = nbr_fea.reshape(-1, 1)


        atom_coords = [site.coords for site in crystal.sites]  # 计算最小距离矩阵
        n_atoms = len(atom_coords)
        lattice_matrix = crystal.lattice.matrix
        min_dist = np.zeros((n_atoms, n_atoms))

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = distance(atom_coords[i], atom_coords[j], lattice_matrix)
                min_dist[i][j] = dist
                min_dist[j][i] = dist

        # 找到矩阵中的零元素
        zero_indices = np.where(min_dist == 0)

        # 对矩阵所有非零元素取倒数
        reciprocal_min_dist = np.zeros_like(min_dist)
        non_zero_indices = np.where(min_dist != 0)
        reciprocal_min_dist[non_zero_indices] = 1 / min_dist[non_zero_indices]
        reciprocal_min_dist = reciprocal_min_dist * 2
        reciprocal_min_dist = np.array(reciprocal_min_dist).T.astype(np.float32)

        g = Data(x=atom_features, edge_index=nbr_fea_idx, edge_attr=nbr_fea, adj=reciprocal_min_dist)
        # Save the dictionary to a pickle file
        with open(filename, 'wb') as f:
            pickle.dump(g, f)
            print(crystal_id)
        features = PygStructureDataset._get_attribute_lookup("cgcnn")
        g.x, g.edge_index, g.edge_attr, g.adj = torch.tensor(g.x), torch.tensor(
            g.edge_index), torch.tensor(g.edge_attr), torch.tensor(g.adj)
        z = g.x
        g.atomic_number = z
        z = z.type(torch.IntTensor).squeeze()
        f = torch.tensor(features[z]).type(torch.FloatTensor)
        if g.x.size(0) == 1:
            f = f.unsqueeze(0)
        g.x = f
        # 设置边特征和邻接矩阵
        g.edge_attr = g.edge_attr.float()
        g.adj = torch.mm(g.adj, g.x) / 1000
        g.x = g.x + g.adj
        feature_cs = Data()
        feature_cs.x, feature_cs.edge_index, feature_cs.edge_attr = g.x, g.edge_index, g.edge_attr

        return feature_cs

    feature = df['crystal_id'].apply(extract_features_cif).values
    num_osda = df['Loading']
    # print(num_osda.shape)

    for i,j in zip(feature, num_osda):
        i.num_osda = torch.tensor(j)
    print(len(feature))

    # print(len(features_total))
    return feature



# torch config
torch.set_default_dtype(torch.float32)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


dataset = pd.read_csv('005model_new_finally.csv')



df = pd.DataFrame(dataset)
atom_features = "cgcnn"

# std_train = 58.658446
# mean_train = -139.596734
# new
# std_train = 60.018909
# mean_train = -130.734573

#03
# mean_train = -125.514320
# std_train = 61.848293

#04
mean_train = -124.002579
std_train = 61.241516

# 05
# mean_train = -120.952515
# std_train = 59.570572


# t_start = time.time()
graphs = load_pyg_graphs(
    df,
    name="dft_3d",
    neighbor_strategy="k-nearest",
    cutoff=4,
    max_neighbors=12,
    use_canonize=False,
    use_lattice=False,
    use_angle=False,
)
# t_end = time.time()
# print("Load time:", t_end - t_start)
data = PygStructureDataset(
    df,
    graphs,
    target='Binding_OSDA',
    atom_features=atom_features,
    line_graph=False,
    id_tag="crystal_id",
    classification=False,
    neighbor_strategy='k-nearest',
    mean_train=mean_train,
    std_train=std_train,
)

collate_fn = data.collate

test_loader = DataLoader(
    data,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=False,
    #    num_workers=workers,
    #    pin_memory=pin_memory,
)

t1 = time.time()
result = train_zeo(config, test_loader=test_loader, test_only=True, prepare_batch=test_loader.dataset.prepare_batch)
t2 = time.time()
print("Toal time:", t2 - t1)
print()
