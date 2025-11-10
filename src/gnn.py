# src/gnn.py
import torch
import torch.nn.functional as F
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import dense_to_sparse
    PYGEOM_OK = True
except Exception:
    PYGEOM_OK = False

class SectorGNN(torch.nn.Module):
    def __init__(self, in_feat, hidden=32, out_feat=1):
        super().__init__()
        if not PYGEOM_OK:
            raise RuntimeError("torch_geometric not available. Skip GNN.")
        self.conv1 = GCNConv(in_feat, hidden)
        self.conv2 = GCNConv(hidden, out_feat)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)
