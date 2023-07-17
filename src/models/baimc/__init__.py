import logging
from pathlib import Path as P
from typing import List
import random

import numpy as np
import torch
from sklearn.preprocessing import normalize

from src.data.dataset import PartialMultiviewDataset
from src.utils.io_utils import train_begin, train_end, save_var
from src.utils.torch_utils import *
from src.utils.metrics import Evaluate_Graph, MaxMetrics, KMeans_Evaluate
from sklearn.cluster import KMeans
from src.vis.visualize import *


class Model_SubspaceAnchor_SVC(nn.Module):
    """
    子空间锚点单视角模型。
    """

    def __init__(
        self,
        U,
        A,
        lr: float = 0.1,
        epochs: int = 100,
        valid_freq=10,
        verbose=False,
    ):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.valid_freq = valid_freq
        self.verbose = verbose
        self.U = U
        self.A = A
        self.sampleNum, self.anchorNum = U.size(0), A.size(0)
        self.params = nn.Parameter(torch.empty(self.sampleNum, self.anchorNum))
        nn.init.normal_(self.params)

    def forward_subspace(self):
        """
        计算子空间稀疏系数矩阵。
        """
        logits = torch.exp(self.params)
        normalization = EPS_max(logits.sum(1)).unsqueeze(1)
        Z = logits / normalization
        return Z

    def forward(self):
        Z = self.forward_subspace()
        xbar = Z @ self.A
        loss = F.mse_loss(xbar, self.U)
        return loss

    def fit(self):
        """
        训练子空间学习模型，学到一个概率图S，本身可直接用于谱聚类，
        也可以作为PTSNE的输入图，进一步学习，提升性能。
        """
        config = self
        model = self
        optim = Adam(model.parameters(), lr=config.lr)

        for epoch in range(config.epochs):
            loss = model.forward()
            optim.zero_grad()
            loss.backward()
            optim.step()
            if (1 + epoch) % config.valid_freq == 0 and config.verbose:
                logging.info(f"epoch {epoch:04d} loss {loss:.6f}")
        # done
        with torch.no_grad():
            model.eval()
            self.Z = model.forward_subspace()


def learn_anchor_graph(A: np.ndarray, U: np.ndarray, device: int):
    A = convert_tensor(A, dev=device)
    U = convert_tensor(U, dev=device)
    model = Model_SubspaceAnchor_SVC(A=A, U=U).to(device)
    model.fit()
    Z = convert_numpy(model.Z)
    return Z


def fuse_incomplete_view_z(
    Z: List[Tensor],
    W: List[Tensor],
    output_shape: tuple,
):
    """
    将按照[A, U]排列的局部锚点图融合为全局锚点图（按原始样本顺序）
    """
    device = Z[0].device
    # 为了节省内存，分配一个原地加法的内存。
    numerator = torch.zeros(output_shape, device=device)
    dominator = torch.zeros(output_shape, device=device)
    for v in range(len(Z)):
        numerator[W[v]] += Z[v]
        dominator[W[v]] += 1

    # 除零错误处理。
    zero_places = dominator == 0.0
    assert torch.all(numerator[zero_places] == 0.0)
    dominator[dominator == 0] = 1  # 如果有0，说明分子也是零.

    Z_fused = numerator / dominator
    return Z_fused


def view_specific_bagging(X, num_anchors, device: int, t: int):
    n = X.shape[0]
    Zv = np.zeros((n, num_anchors))
    Dv = np.zeros(n)

    # bagging
    for i in range(t):
        anchor_idx = random.sample(range(n), k=num_anchors)
        anchor_mask = np.zeros(n, dtype=bool)
        anchor_mask[anchor_idx] = 1
        sample_mask = ~anchor_mask
        A = X[anchor_mask]
        U = X[sample_mask]
        Zi = learn_anchor_graph(A=A, U=U, device=device)
        Zv[sample_mask] += Zi
        Dv[sample_mask] += 1

    Zv = Zv / np.expand_dims(Dv, 1)
    return Zv


def view_specific_kmeans(X, num_anchors, device: int):
    kmeans = KMeans(n_clusters=num_anchors)
    X = normalize(X)
    kmeans.fit(X)
    A = kmeans.cluster_centers_
    U = X
    Z = learn_anchor_graph(A, U, device=device)
    return Z


def train_main(
    datapath=P("./data/ORL-40.mat"),
    eta=0.5,
    views=None,
    k: int = 5,
    t1: int = 20,
    t2: int = 20,
    view_graph: str = "bagging",
    device="cpu",
    savedir: P = P("output/debug"),
    save_vars: bool = False,
):
    # if 'ORL' in datapath.name:
    #     k = 3
    #     t = 20
    #     t2 = 10
    # elif 'Digits' in datapath.name:
    #     k = 5
    # elif 'COIL' in datapath.name:
    #     k = 5
    #     t = 10
    #     t2 = 20

    mm = MaxMetrics()
    method_name = f"BAIMC"
    config = dict(
        datapath=datapath,
        eta=eta,
        views=views,
        method=method_name,
        k=k,
        t1=t1,
        t2=t2,
    )
    train_begin(savedir, config, f"Begin train {method_name}")

    data = PartialMultiviewDataset(
        datapath=datapath,
        paired_rate=1 - eta,
        view_ids=views,
        normalize="minmax",
    )

    print(data.describe())

    X_all = data.X_all_list
    idx_all = data.idx_all_list
    num_anchors = int(k * data.clusterNum)

    history = []
    Z_common = torch.zeros(data.sampleNum, num_anchors)
    H_common = None

    for j in range(t2):
        Z = [None] * data.viewNum
        for v in range(data.viewNum):
            if view_graph == "kmeans":
                Z[v] = view_specific_kmeans(
                    X_all[v], num_anchors=num_anchors, device=device
                )
            elif view_graph == "bagging":
                Z[v] = view_specific_bagging(
                    X_all[v], num_anchors=num_anchors, device=device, t=t1
                )

        Z = convert_tensor(Z, dev=device)
        Z_common += fuse_incomplete_view_z(
            Z=Z, W=idx_all, output_shape=(data.sampleNum, num_anchors)
        )
        Z_temp = Z_common / (j + 1)
        metrics, ff = Evaluate_Graph(
            data=data, Z=Z_temp, type="fastSVD", return_spectral_embedding=True
        )
        print(f"epoch {j:02} metrics {metrics}")
        history.append(metrics)
        if mm.update(**metrics)["ACC"] and save_vars:
            H_common = ff

    train_end(savedir, mm.report(current=False))
    if save_vars:
        save_var(savedir, H_common, "H_common")
        save_var(savedir, history, "history")
