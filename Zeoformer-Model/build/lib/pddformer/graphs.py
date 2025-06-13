"""Module to generate networkx graphs."""
"""Implementation based on the template of ALIGNN, Matformer."""
from multiprocessing.context import ForkContext
from re import X
import numpy as np
import pandas as pd
from jarvis.core.specie import chem_data, get_node_attributes

# from jarvis.core.atoms import Atoms
from collections import defaultdict
from typing import List, Tuple, Sequence, Optional
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from torch_geometric.data.batch import Batch
import itertools

try:
    import torch
    from tqdm import tqdm
except Exception as exp:
    print("torch/tqdm is not installed.", exp)
    pass

# pyg dataset
class PygStructureDataset(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
        self,
        df: pd.DataFrame,
        graphs: Sequence[Data],
        target: str,
        atom_features="atomic_number",
        transform=None,
        classification=False,
        id_tag="jid",
        neighbor_strategy="",
        mean_train=None,
        std_train=None,
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        `graphs`: DGLGraph representations corresponding to rows in `df`
        `target`: key for label column in `df`
        """
        self.df = df
        self.graphs = graphs
        self.target = target

        self.ids = self.df[id_tag]
        self.atoms = self.df['atoms']
        self.labels = torch.tensor(self.df[target]).type(
            torch.get_default_dtype()
        )
        print("mean %f std %f"%(self.labels.mean(), self.labels.std()))
        if mean_train == None:
            mean = self.labels.mean()
            std = self.labels.std()
            self.labels = (self.labels - mean) / std
            print("normalize using training mean but shall not be used here %f and std %f" % (mean, std))
        else:
            self.labels = (self.labels - mean_train) / std_train
            print("normalize using training mean %f and std %f" % (mean_train, std_train))

        self.transform = transform

        features = self._get_attribute_lookup(atom_features)

        # load selected node representation
        # assume graphs contain atomic number in g.ndata["atom_features"]
        for g in graphs:
            z = g.node
            g.atomic_number = z
            z = z.type(torch.IntTensor).squeeze()
            f = torch.tensor(features[z]).type(torch.FloatTensor)
            if g.node.size(0) == 1:
                f = f.unsqueeze(0)
            g.node = f
        self.prepare_batch = prepare_pyg_batch
        self.graphs = []
        for g in tqdm(graphs):
            g.edge_attr = g.edge_attr.float()
            # g.pdd = torch.mm(g.pdd, g.node)
            self.graphs.append(g)
        self.line_graphs = self.graphs


        if classification:
            self.labels = self.labels.view(-1).long()
            print("Classification dataset.", self.labels)

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        """Build a lookup array indexed by atomic number."""
        max_z = max(v["Z"] for v in chem_data.values())

        # get feature shape (referencing Carbon)
        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features

    def __len__(self):
        """Get length."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        g = self.graphs[idx]
        label = self.labels[idx]

        if self.transform:
            g = self.transform(g)

        return g, label

    def setup_standardizer(self, ids):
        """Atom-wise feature standardization transform."""
        x = torch.cat(
            [
                g.node
                for idx, g in enumerate(self.graphs)
                if idx in ids
            ]
        )
        self.atom_feature_mean = x.mean(0)
        self.atom_feature_std = x.std(0)

        self.transform = PygStandardize(
            self.atom_feature_mean, self.atom_feature_std
        )

    @staticmethod
    def collate(samples: List[Tuple[Data, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        return batched_graph, torch.tensor(labels)


def nearest_neighbor_edges_submit(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    id=None,
    use_canonize=False
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    lat = atoms.lattice
    # print(lat[0][0])
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    attempt = 0
    if min_nbrs < 50:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1
        return nearest_neighbor_edges_submit(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
        )

    atom_coords = atoms.coords
    n_atoms = len(atom_coords)
    min_dist = np.zeros((n_atoms, 50))

    # PDD Information Extraction: Selecting 92 Edges
    for i in range(n_atoms):
        neighborlist = all_neighbors[i]  # 访问第 i 个原子的邻居列表
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        # ids = np.array([nbr[1] for nbr in neighborlist])
        dis = np.array([nbr[2] for nbr in neighborlist])
        # Calculate the Row PDD for Each Element
        min_dist[i][:] = dis[:50]
        # atomic_index[i][:] = ids[:92]

    # Take the Reciprocal of the PDD Matrix
    # reciprocal_min_dist = np.zeros_like(min_dist)
    # non_zero_indices = np.where(min_dist != 0)
    # reciprocal_min_dist[non_zero_indices] = 1 / min_dist[non_zero_indices]
    reciprocal_min_dist = np.array(min_dist).astype(np.float32)
    reciprocal_min_dist = torch.tensor(reciprocal_min_dist)

    # edges = defaultdict(set)
    edge_index = []
    edge_ettr = []

    for site_idx, neighborlist in enumerate(all_neighbors):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]


        # Neighbor Selection After Introducing Continuous Tolerance T
        # distances = distances[distances <= max_dist]
        # for i in range(max_neighbors, len(distances)):
        #     if distances[i] - distances[i-1] > 0.001/max(lat.a, lat.b, lat.c):
        #         max_dist = distances[i-1]
        #         break


        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]
        for dst, dis in zip(ids, distances):
            edge_index.append((site_idx, dst))
            edge_ettr.append(dis)


    edge_index = np.array(edge_index).astype(np.int64)
    edge_ettr = np.array(edge_ettr)
    edge_ettr = edge_ettr.astype(np.float32)
    edge_index = np.transpose(edge_index)
    edge_index = torch.tensor(edge_index)
    edge_ettr = torch.tensor(edge_ettr)
    edge_ettr = edge_ettr.unsqueeze(1)
    return edge_index, edge_ettr, reciprocal_min_dist


def distance(pos1, pos2, lattice_matrix):
    """Calculate the Distance Between Two Atoms Considering Lattice Parameters"""
    delta = pos2 - pos1
    delta = np.dot(delta, np.linalg.inv(lattice_matrix))
    delta -= np.round(delta)
    delta = np.dot(delta, lattice_matrix)
    dist = np.linalg.norm(delta)
    return dist


class PygGraph(object):
    """Generate a graph object."""

    def __init__(
        self,
        nodes=[],
        node_attributes=[],
        edges=[],
        edge_attributes=[],
        color_map=None,
        labels=None,
    ):
        """
        Initialize the graph object.

        Args:
            nodes: IDs of the graph nodes as integer array.

            node_attributes: node features as multi-dimensional array.

            edges: connectivity as a (u,v) pair where u is
                   the source index and v the destination ID.

            edge_attributes: attributes for each connectivity.
                             as simple as euclidean distances.
        """
        self.nodes = nodes
        self.node_attributes = node_attributes
        self.edges = edges
        self.edge_attributes = edge_attributes
        self.color_map = color_map
        self.labels = labels

    @staticmethod
    def atom_dgl_multigraph(
        atoms=None,
        neighbor_strategy="k-nearest",
        cutoff=8.0,
        max_neighbors=12,
        atom_features="cgcnn",
        max_attempts=3,
        id: Optional[str] = None,
        use_canonize: bool = False,
    ):
        # Below is a Python example code snippet for UPDD extraction
        # lattice_matrix = atoms.lattice_mat
        # atom_coords = atoms.coords
        #
        # n_atoms = len(atom_coords)
        # min_dist = np.zeros((n_atoms, n_atoms))
        #
        # for i in range(n_atoms):
        #     for j in range(i + 1, n_atoms):
        #         dist = distance(atom_coords[i], atom_coords[j], lattice_matrix)
        #         min_dist[i][j] = dist
        #         min_dist[j][i] = dist
        #
        # # Find Zero Elements in UPDD Matrix
        # zero_indices = np.where(min_dist == 0)
        #
        # # Take the Reciprocal of the UPDD Matrix
        # reciprocal_min_dist = np.zeros_like(min_dist)
        # non_zero_indices = np.where(min_dist != 0)
        # reciprocal_min_dist[non_zero_indices] = 1 / min_dist[non_zero_indices]
        # reciprocal_min_dist = reciprocal_min_dist
        # reciprocal_min_dist = np.array(reciprocal_min_dist).T.astype(np.float32)
        # reciprocal_min_dist = torch.tensor(reciprocal_min_dist)


        if neighbor_strategy == "k-nearest":
            edge_index, r, reciprocal_min_dist = nearest_neighbor_edges_submit(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                id=id,
                use_canonize=use_canonize
            )
        else:
            raise ValueError("Not implemented yet", neighbor_strategy)
        # edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()

        """
        edge_index, r = crystal_diatance(atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,)
        print(edge_index)
        print(r)
        print("------------")
        """

        # build up atom attribute tensor
        sps_features = []
        for ii, s in enumerate(atoms.elements):
            feat = list(get_node_attributes(s, atom_features=atom_features))
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )

        # print(len(node_features))
        total_sum = node_features.sum().float()

        # Calculate the Proportion of Each Element to the Total Sum
        atomic_weight = node_features.float() / total_sum
        reciprocal_min_dist = np.hstack((atomic_weight, reciprocal_min_dist))
        reciprocal_min_dist = torch.tensor(reciprocal_min_dist)

        g = Data(node=node_features, edge_index=edge_index, edge_attr=r, pdd=reciprocal_min_dist)

        return g

class PygStandardize(torch.nn.Module):
    """Standardize atom_features: subtract mean and divide by std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Register featurewise mean and standard deviation."""
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, g: Data):
        """Apply standardization to atom_features."""
        h = g.node
        g.node = (h - self.mean) / self.std
        return g



def prepare_pyg_batch(
    batch: Tuple[Data, torch.Tensor], device=None, non_blocking=False
):
    """Send batched dgl crystal graph to device."""
    g, t = batch
    batch = (
        g.to(device),
        t.to(device, non_blocking=non_blocking),
    )

    return batch


