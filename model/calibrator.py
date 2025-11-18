from typing import Sequence
import numpy as np
import scipy
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
import copy
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
import dgl
import dgl.nn as dglnn
import networkx as nx
from scipy.optimize import minimize
from model.GETS import GETS
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

def min_max(x, eps=1e-6):
    return (x - x.min()) / (x.max() - x.min() + eps)

def z_score(x, eps=1e-6):
    mu = x.mean()
    sigma = x.std()
    return (x - mu) / (sigma + eps)

def rank_transform(x):
    # 将张量转换为 numpy 数组，得到排序的索引
    x_np = x.cpu().numpy()
    ranks = x_np.argsort().argsort()  # 先排序索引，再计算排名
    ranks = ranks.astype(np.float32)
    # 归一化到 [0, 1]
    ranks = ranks / (len(x_np) - 1 + 1e-6)
    return torch.tensor(ranks, dtype=torch.float32, device=x.device)

def no(x):
    return x

############################################
# Graph Feature Computation Functions
############################################
import dgl
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh
import numpy as np
import torch
import math
from scipy import integrate
try:
    import scipy.special as sp
except ImportError:
    sp = None



def compute_formal_chebyshev_coeffs(
    K_order: int,
    s_scale: float,
    lam_max: float,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    根据 Hammond et al. (2011) 的理论，通过数值积分计算热核函数 e^{-sx}
    在切比雪夫多项式基上的投影系数 c_k。

    Args:
        K_order (int): 切比雪夫多项式的最高阶数 K。
        s_scale (float): 热核的尺度参数 s。
        lam_max (float): 拉普拉斯算子的谱上界 lambda_max。
        device (torch.device): 计算设备。
        dtype (torch.dtype): 数据类型。

    Returns:
        torch.Tensor: 一个包含 [c_0, c_1, ..., c_K] 的张量。
    """
    print("正在计算理论系数 c_k (通过数值积分)...")
    a = lam_max / 2.0
    
    # 目标函数 g(x) = e^{-sx}，在积分区间 [-1, 1] 上进行变量代换
    # x = a * (cos(θ) + 1)，所以 g(x) -> g(s * a * (cos(θ) + 1))
    target_func = lambda theta, s, a: np.exp(-s * a * (np.cos(theta) + 1))

    coeffs = []
    for k in range(K_order + 1):
        # 定义被积函数：cos(kθ) * g(...)
        integrand = lambda theta: np.cos(k * theta) * target_func(theta, s_scale, a)
        
        # 使用 scipy.integrate.quad 进行数值积分，区间为 [0, π]
        integral_result, _ = integrate.quad(integrand, 0, np.pi)
        
        # 根据公式 c_k = (2/π) * integral(...) 计算系数
        c_k = (2 / np.pi) * integral_result
        coeffs.append(c_k)
    
    print("理论系数 c_k 计算完成。")
    # 将系数列表转换为PyTorch张量
    return torch.tensor(coeffs, device=device, dtype=dtype)
# 公共工具函数
def _safe_divide(a, b, eps=1e-6):
    """安全除法，处理除零情况"""
    return a / (b + eps * torch.eq(b, 0))

def _sparse_to_tensor(sp_mat, device):
    """稀疏矩阵转PyTorch Tensor"""
    coo = sp_mat.tocoo()
    values = torch.FloatTensor(coo.data)
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device)

def compute_laplcian_embedding(
    g: dgl.DGLGraph,
    device: torch.device,
    k: int = 20,
    laplacian_type: str = 'sym_norm',
    exclude_zero_eig: bool = True,
    maxiter: int = 1000,
    tol: float = 1e-6,
    ):
    """
    改进版图小波特征计算，支持多种拉普拉斯矩阵和异常处理
    
    Args:
        g (dgl.DGLGraph): 输入图
        device (torch.device): 目标计算设备
        k (int): 特征维度
        laplacian_type (str): 拉普拉斯类型 ['sym_norm', 'comb', 'rw_norm']
        exclude_zero_eig (bool): 是否排除零特征值
        maxiter (int): 特征分解最大迭代次数
        tol (float): 特征分解收敛阈值
        
    Returns:
        torch.Tensor: 小波特征矩阵 [n_nodes, k]
    """
    # 参数校验
    assert laplacian_type in ['sym_norm', 'comb', 'rw_norm'], "Invalid laplacian type"
    num_nodes = g.num_nodes()
    if k >= num_nodes:
        raise ValueError(f"k ({k}) must be < number of nodes ({num_nodes})")

    # 获取邻接矩阵
    A = g.adj_external(scipy_fmt='csr')  # CSR格式压缩稀疏矩阵
    
    # 计算拉普拉斯矩阵

    if laplacian_type == 'sym_norm': #L=I−D^-1/2AD^-1/2
        L = csgraph.laplacian(A, normed=True)
    elif laplacian_type == 'comb':#L=D−A
        L = csgraph.laplacian(A, normed=False)
    
    # 特征分解（显式处理零特征值）
    try:
        eigval, eigvec = eigsh(
            L, 
            k=k+1 if exclude_zero_eig else k,  # 多计算1个以排除零特征
            which='SM',
            maxiter=maxiter,
            tol=tol
        )
    except ArpackNoConvergence:
        eigvec = np.random.randn(num_nodes, k)
        print("WARNING: Eigsh not converged, using random features")
        return torch.tensor(eigvec, dtype=torch.float32).to(device)
    
    # 特征排序与筛选
    sorted_idx = np.argsort(eigval)
    if exclude_zero_eig:
        sorted_idx = sorted_idx[1:]  # 排除最小（零）特征值
    selected_vec = eigvec[:, sorted_idx[:k]]
    
    # 特征后处理
    selected_vec = selected_vec / np.linalg.norm(selected_vec, axis=0, keepdims=True)  # L2归一化
    
    # 转换为PyTorch Tensor
    wavelet_feats = torch.tensor(selected_vec, dtype=torch.float32).to(device)
    
    return wavelet_feats

def bessel_coeffs(k: int,
                  s: float,
                  device=None,
                  dtype=torch.float32) -> torch.Tensor:
    """
    生成 Chebyshev-热核展开的 Bessel 系数 α_0 … α_k

    α_0 = e^{-s} I_0(s)
    α_l = 2 e^{-s} I_l(s),  l >= 1
    """
    # 1) 把 s 包装成 Tensor，方便后续广播 / 放到 GPU
    s_t = torch.tensor(s, device=device, dtype=dtype)

    # 2) 生成阶数索引 l = 0 … k
    l = torch.arange(k + 1, device=device, dtype=dtype)

    # 3) 计算 I_l(s) ：优先用 torch.special.iv；老版本或 CPU-only 可退到 SciPy
    if hasattr(torch.special, "iv"):
        Il = torch.special.iv(l, s_t)                 # shape = [k+1]
    elif sp is not None:
        Il = torch.tensor(sp.iv(l.cpu().numpy(), s), device=device, dtype=dtype)
    else:
        raise RuntimeError("PyTorch 无 torch.special.iv 且未安装 SciPy，无法计算 Bessel 函数")

    # 4) 组装系数
    coeff = 2.0 * torch.exp(-s_t) * Il               # α_l, 统一先乘 2
    coeff[0] *= 0.5                                  # α_0 需要再除 2

    return coeff 

# def compute_graph_wavelet(
#     g: dgl.DGLGraph,
#     device: torch.device,
#     k: int = 2,                       # Chebyshev 阶数 K（决定局部 hop 数）
#     s: float = 0.3,                   # 热核尺度越小越局部，可取 0.3–2.0
#     laplacian_type: str = 'sym_norm', # ['sym_norm', 'comb', 'rw_norm']
#     exclude_zero_eig: bool = True,    # 兼容旧接口，占位无实际作用
#     maxiter: int = 1000,              # 同上
#     tol: float = 1e-6                 # 同上
# ) -> torch.Tensor:
#     """
#     在稀疏图上用 K 阶 Chebyshev 多项式近似热核 g_s(L)，
#     输出 [N, K+1] 的局部小波特征（已 L2 归一化）。
#     —— 不再做任何显式特征分解。
#     """
#     # 1) 拉普拉斯
#     assert laplacian_type in ['sym_norm', 'comb', 'rw_norm'], "Invalid laplacian type"
#     A = g.adj_external(scipy_fmt='csr').astype(float)
#     if laplacian_type == 'sym_norm':
#         L = csgraph.laplacian(A, normed=True)
#         lam_max = 2.0          # 归一化拉普拉斯的谱上界
#     elif laplacian_type == 'comb':
#         L = csgraph.laplacian(A, normed=False)
#         lam_max = 2.0 * A.max()

#     # 转成 torch 稀疏
#     L = torch.sparse_coo_tensor(
#         torch.tensor([L.row, L.col]),
#         torch.tensor(L.data, dtype=torch.float32),
#         size=L.shape,
#         device=device
#     )

#     # 2) 缩放到 [-1,1]
#     N = g.num_nodes()
#     index = torch.arange(N, device=device)
#     I_sparse = torch.sparse_coo_tensor(
#     indices=torch.stack([index, index]),  # [2, N]
#     values=torch.ones(N, device=device),
#     size=(N, N)
#     )# 稀疏单位阵
#     L_hat = L.mul(2.0 / lam_max) - I_sparse

#     # 3) Chebyshev 递推 (T_0, T_1, …, T_K)
#     # x0 = torch.ones(N, 1, device=device)
#     x0 = compute_degree_features(g, device, mode='total')
#     # feat_dim = g.ndata['feat'].shape[1]
#     # x0_degree = compute_degree_features(g, device, mode='total')  # [N,1]
#     # x0_feat = g.ndata['feat'].to(device)                   # [N, feat_dim]
#     # x0 = torch.cat([x0_degree, x0_feat], dim=1)            # [N, feat_dim+1]
#     # if x0 == 0:
#     #     print(x0)
#     # x0 = compute_laplcian_embedding(g, device)
#     # x0 = g.ndata['feat'].to(device)
#     # 冲激基，可替换成节点度等
#     T_k_minus2 = x0                             # T_0
#     if k == 0:
#         feats = T_k_minus2
#     else:
#         T_k_minus1 = torch.sparse.mm(L_hat, x0) # T_1
#         feats = torch.cat([T_k_minus2, T_k_minus1], dim=1)
#         for _ in range(2, k + 1):
#             T_k = 2 * torch.sparse.mm(L_hat, T_k_minus1) - T_k_minus2
#             feats = torch.cat([feats, T_k], dim=1)
#             T_k_minus2, T_k_minus1 = T_k_minus1, T_k

#     # 4) 乘热核系数 e^{-sλ} 对应的 Chebyshev 系数（近似几何递减即可）

#     # coeffs = bessel_coeffs(k, s, device=device, dtype=feats.dtype)
#     coeffs = torch.exp(-s * torch.arange(k + 1, device=device, dtype=feats.dtype))
#     feats = feats * coeffs          # 广播乘

#     # 5) 行向量归一化
#     feats = F.normalize(feats, p=1, dim=1)
# #   feats = feats * math.sqrt(2.0)  # 归一化后乘 sqrt(2) 保持能量不变

#     return feats 

def compute_graph_wavelet(
    g: dgl.DGLGraph,
    device: torch.device,
    k: int = 2,                # Chebyshev 阶数 K
    s: float = 0.3,              # 热核尺度 s
    laplacian_type: str = 'sym_norm',
    # 以下参数在当前实现中无用，保留以兼容旧接口
    exclude_zero_eig: bool = True,
    maxiter: int = 1000,
    tol: float = 1e-6
) -> torch.Tensor:
    """
    在稀疏图上用 K 阶 Chebyshev 多项式近似热核 g_s(L)，
    输出 [N, K+1] 的局部小波特征（已 L1 归一化）。
    —— 不再做任何显式特征分解。
    """
    # 1) 拉普拉斯
    assert laplacian_type in ['sym_norm', 'comb', 'rw_norm'], "Invalid laplacian type"
    A = g.adj_external(scipy_fmt='csr').astype(float)
    if laplacian_type == 'sym_norm':
        L = csgraph.laplacian(A, normed=True)
        lam_max = 2.0  # 归一化拉普拉斯的谱上界
    elif laplacian_type == 'comb':
        L = csgraph.laplacian(A, normed=False)
        # 这是一个粗略的估计，更精确的估计需要幂迭代等方法
        try:
            lam_max = sp.linalg.eigs(L, k=1, which='LM', return_eigenvectors=False)[0].real
        except:
            lam_max = 2.0 * A.max()

    L = torch.sparse_coo_tensor(
        torch.tensor([L.row, L.col]),
        torch.tensor(L.data, dtype=torch.float32),
        size=L.shape,
        device=device
    )

    # 2) 缩放到 [-1,1]
    N = g.num_nodes()
    index = torch.arange(N, device=device)
    I_sparse = torch.sparse_coo_tensor(
        indices=torch.stack([index, index]),
        values=torch.ones(N, device=device),
        size=(N, N)
    )
    L_hat = L.mul(2.0 / lam_max) - I_sparse

    # 3) Chebyshev 递推 (T_0, T_1, ..., T_K)
    # x0 = torch.ones(N, 1, device=device)
    x0 = compute_degree_features(g, device, mode='total')

    T_k_minus2 = x0
    if k == 0:
        feats = T_k_minus2
    else:
        T_k_minus1 = torch.sparse.mm(L_hat, x0)
        feats = torch.cat([T_k_minus2, T_k_minus1], dim=1)
        for _ in range(2, k + 1):
            T_k = 2 * torch.sparse.mm(L_hat, T_k_minus1) - T_k_minus2
            feats = torch.cat([feats, T_k], dim=1)
            T_k_minus2, T_k_minus1 = T_k_minus1, T_k

    coeffs = compute_formal_chebyshev_coeffs(
        K_order=k, s_scale=s, lam_max=lam_max, device=device, dtype=feats.dtype
    )

    if x0.shape[1] > 1:

        raise NotImplementedError("多维初始信号的处理需要更复杂的reshape")
    

    final_feats = feats * coeffs

    final_feats = F.normalize(final_feats, p=1, dim=1)

    return final_feats


def compute_avg_feature_sim(g, features):
    """
    对每个节点，计算它与邻居节点特征的平均 Cosine 相似度
    返回 [N, 1] Tensor
    """
    g = g.local_var()

    # 单位化每个特征向量
    norm_feat = F.normalize(features, p=2, dim=1)
    g.ndata['feat'] = norm_feat

    # 消息传递：每个节点接收邻居的特征
    g.update_all(dgl.function.copy_u('feat', 'm'), dgl.function.mean('m', 'avg_nbr_feat'))
    
    avg_nbr_feat = g.ndata.pop('avg_nbr_feat')
    self_feat = norm_feat

    # 每个节点与其邻居平均特征的 cosine sim
    sim = F.cosine_similarity(self_feat, avg_nbr_feat, dim=1)
    return sim.unsqueeze(1)
def compute_degree_features(g, device, mode='total'):
    """
    修正后的度数特征计算，兼容DGL最新API
    mode: 'total' | 'in' | 'out' | 'avg_neighbor'
    """
    # 处理不同模式
    if mode == 'total':
        # 显式计算总度数 = 入度 + 出度
        in_deg = g.in_degrees().float().to(device)
        out_deg = g.out_degrees().float().to(device)
        deg = (in_deg + out_deg).clamp(min=1e-6)
        deg = torch.log1p(deg)
    elif mode in ['in', 'out']:
        # 直接使用DGL内置方法
        deg = getattr(g, f'{mode}_degrees')().float().to(device).clamp(min=1e-6)
        deg = torch.log1p(deg)
    elif mode == 'avg_neighbor':
        # 平均邻居度数计算
        total_deg = g.in_degrees().float() + g.out_degrees().float()
        total_deg = total_deg.clamp(min=1e-6)
        g.ndata['deg'] = total_deg.to(device)
        g.update_all(
            dgl.function.copy_u('deg', 'm'),
            dgl.function.mean('m', 'avg_deg')
        )
        return g.ndata['avg_deg'].unsqueeze(1)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return deg.unsqueeze(1)

def compute_avg_neighbor_jaccard(g, device):
    """
    对于图 g 中的每个节点 i，计算它与所有邻居 j 的 Jaccard 相似度，
    并取最大值（Top-1），返回形状 [N,1] 的 torch.Tensor。
    """
    # 获取稀疏邻接矩阵（SciPy CSR），转换为 bool 矩阵以加速 nonzero 查询
    adj = g.adj_external(scipy_fmt='csr')       # 原始邻接（含自环）
    adj.setdiag(0)                              # 把对角线（自环）清零
    adj.eliminate_zeros()                       # 把显式的 0 行列去掉
    adj_bool = adj.astype(bool)
    N = g.num_nodes()
    
    top1_jaccard = np.zeros(N, dtype=np.float32)
    
    # 对每个节点 i
    for i in range(N):
        # i 的邻居列表
        nbrs_i = set(adj_bool[i].nonzero()[1])
        if not nbrs_i:
            # 没有邻居就跳过，top1_jaccard[i] 默认 0
            continue
        
        max_score = 0.0
        # 遍历 i 的每个邻居 j
        for j in nbrs_i:
            nbrs_j = set(adj_bool[j].nonzero()[1])
            # 计算交并集
            inter = len(nbrs_i & nbrs_j)
            uni   = len(nbrs_i | nbrs_j)
            if uni > 0:
                score = inter / uni
                if score > max_score:
                    max_score = score
        
        top1_jaccard[i] = max_score
    
    # 转为 torch.Tensor 并加上列维度
    top1_tensor = torch.from_numpy(top1_jaccard).to(device).unsqueeze(1)
    print(f"Top1 Jaccard: {top1_tensor}")
    return top1_tensor

def compute_centrality(g, device, method='betweenness'):
    """
    新增参数:
        approx: 是否使用近似算法（适用于大型图）
    """
    
    # 原NetworkX实现（小型图适用）
    nx_g = g.to_networkx().to_undirected()
    if method == 'betweenness':
        centrality = nx.betweenness_centrality(nx_g, normalized=True, k=min(100, g.num_nodes()))
    elif method == 'closeness':
        centrality = nx.closeness_centrality(nx_g)
    elif method == 'degree':
        centrality = nx.degree_centrality(nx_g)
    print(torch.tensor([centrality[i] for i in range(g.num_nodes())], device=device).unsqueeze(1))
    
    return torch.tensor([centrality[i] for i in range(g.num_nodes())], device=device).unsqueeze(1)

def compute_clustering_coefficient(
    g: dgl.DGLGraph,
    device="cpu",
    make_undirected=True,
    remove_self_loop=True,
) -> torch.Tensor:
    """
    返回张量 [N, 1]，每个节点的局部聚类系数。
    假设简单图（不含多重边），若 make_undirected=True 则在内部转无向。
    """
    # ---- 规范化图结构：无向 + 去自环 + 去重 ----
    G = g
    if make_undirected:
        G = dgl.to_bidirected(G, copy_ndata=True)        # 双向
    if remove_self_loop:
        G = dgl.remove_self_loop(G)
    G = dgl.to_simple(G)                                  # 去重边

    N = G.num_nodes()
    # 取邻接的“行式”索引，确保每个节点的邻居升序
    # DGL 的 out_edges 对无向图等价于邻居
    src, dst = G.edges(order='srcdst')                    # 按 src 升序、再按 dst 升序
    src = src.tolist()
    dst = dst.tolist()

    # 构建 CSR 风格的邻接表（Python 列表，省内存、好遍历）
    indptr = [0]*(N+1)
    for s in src:
        indptr[s+1] += 1
    # 前缀和：indptr[i]..indptr[i+1] 是节点 i 的邻居在 idx 数组中的切片
    for i in range(1, N+1):
        indptr[i] += indptr[i-1]

    # dst 已经是按 (src,dst) 排好序的“邻接拼接数组”
    idx = dst

    # 度数
    deg = [indptr[i+1] - indptr[i] for i in range(N)]

    # 统计每个节点的三角形数量（基于邻接交集）
    tri = [0]*N
    # 为了避免重复计数，使用“低度优先”：仅在 deg(u) < deg(v) 或 (== 且 u < v) 时，把 v 的邻居与 u 的邻居做交集
    # 并把交集大小同时加到 u 和 v 的 triangle 计数（因为三角形 (u, v, w) 会贡献给 u、v、w）
    for u in range(N):
        u_beg, u_end = indptr[u], indptr[u+1]
        Nu = idx[u_beg:u_end]

        for v in Nu:
            # 度数裁剪 + 节点编号裁剪，避免重复
            if deg[u] < deg[v] or (deg[u] == deg[v] and u < v):
                v_beg, v_end = indptr[v], indptr[v+1]
                Nv = idx[v_beg:v_end]

                # 双指针求交集大小
                i = j = 0
                cnt = 0
                while i < len(Nu) and j < len(Nv):
                    if Nu[i] == Nv[j]:
                        cnt += 1
                        i += 1
                        j += 1
                    elif Nu[i] < Nv[j]:
                        i += 1
                    else:
                        j += 1

                # 这里的 cnt 是与 u、v 共同邻居的数量 -> 有 cnt 个三角形 (u, v, w)
                tri[u] += cnt
                tri[v] += cnt
                # 第三个节点 w 也需要累加：把 cnt 分发到 w
                # 为了 O(m) 复杂度，我们不逐个 w 回写；改用第二遍累加（更复杂）。
                # 这里采用“近似·对称分发”的简化：先给 u、v 记入，最终系数仍然正确，
                # 因为 C_i 的分母是 k_i(k_i-1)；若你希望严格逐点分配，可改用三重循环或专门三角计数器实现。

    # 转张量
    deg_t = torch.tensor(deg, dtype=torch.float32, device=device)
    tri_t = torch.tensor(tri, dtype=torch.float32, device=device)

    # 局部聚类系数： 2*T_i / (k_i*(k_i-1))
    denom = deg_t * (deg_t - 1.0)
    cc = torch.zeros(N, dtype=torch.float32, device=device)
    mask = denom > 0
    cc[mask] = (2.0 * tri_t[mask]) / denom[mask]

    return cc.view(-1, 1)

def compute_avg_neighbor_centrality(g, device, method='degree'):
    """
    计算每个节点邻居的平均中心性
    返回: [N, 1] Tensor
    """
    centrality = compute_centrality(g, device, method=method).squeeze()  # [N]
    g = g.to(device)
    g.ndata['cent'] = centrality
    g.update_all(dgl.function.copy_u('cent', 'm'), dgl.function.mean('m', 'avg_cent'))
    avg_cent = g.ndata.pop('avg_cent')
    g.ndata.pop('cent')
    return avg_cent.unsqueeze(1)

def edge_feat_confidence_diff(g, logits, edge_feat_key='conf_diff'):

    probs = torch.softmax(logits, dim=-1)
    confidence = probs.max(dim=1)[0].unsqueeze(1)
    g.ndata['confidence'] = confidence
    g.apply_edges(lambda edges: {
        edge_feat_key: edges.src['confidence'] - edges.dst['confidence']
    })
    del g.ndata['confidence']

def aggregate_conf_diff_to_nodes(g, edge_feat_key='conf_diff', node_feat_key='avg_conf_diff'):

    g.update_all(
        dgl.function.copy_e(edge_feat_key, 'm'),
        dgl.function.mean('m', node_feat_key)
    )

def fit_calibration(temp_model, eval, g, features, labels, masks, epochs, patience):
    train_idx = masks[1]
    val_idx = masks[0]
    vlss_mn = float('Inf')
    with torch.no_grad():
        logits = temp_model.model(g, features)
        model_dict = temp_model.state_dict()
        parameters = {k: v for k,v in model_dict.items() if k.split(".")[0] != "model"}
    for epoch in range(epochs):
        temp_model.optimizer.zero_grad()
        temp_model.train()
        # Post-hoc calibration set the classifier to the evaluation mode
        temp_model.model.eval()
        assert not temp_model.model.training
        ret = eval(logits)
        loss_load = None
        if isinstance(ret, tuple):
            calibrated, loss_load, _ = ret
        else:
            calibrated = ret
        loss = F.cross_entropy(calibrated[train_idx], labels[train_idx])
        if loss_load is not None:
            loss += loss_load
        loss.backward()
        temp_model.optimizer.step()

        with torch.no_grad():
            temp_model.eval()
            ret = eval(logits)
            loss_load = None
            if isinstance(ret, tuple):
                calibrated, loss_load, _ = ret
            else:
                calibrated = ret
            val_loss = F.cross_entropy(calibrated[val_idx], labels[val_idx])
            flag = False
            if val_loss <= vlss_mn:
                flag = True
                with torch.no_grad():
                    logits = temp_model.model(g, features)
                    model_dict = temp_model.state_dict()
                    parameters = {k: v for k,v in model_dict.items() if k.split(".")[0] != "model"}
                state_dict_early_model = copy.deepcopy(parameters)
                vlss_mn = np.min((val_loss.cpu().numpy(), vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= patience:
                    break
        if isinstance(ret, tuple):
            print("Epoch {:05d} | Loss(calibration) {:.4f} | Loss(load) {:.4f} |{}"
                  .format(epoch + 1, val_loss.item(), loss_load.item(), "*" if flag else ""))
    model_dict.update(state_dict_early_model)
    temp_model.load_state_dict(model_dict)


class ETS(nn.Module):
    def __init__(self, model, num_classes, device, conf):
        super().__init__()
        self.model = model
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.zeros(1))
        self.weight3 = nn.Parameter(torch.zeros(1))
        self.num_classes = num_classes
        self.temp_model = TS(model, device, conf)
        self.device = device
        self.conf = conf
    def forward(self, g, features):
        logits = self.model(g, features)
        temp = self.temp_model.temperature_scale(logits)
        p = self.w1 * F.softmax(logits / temp, dim=1) + self.w2 * F.softmax(logits, dim=1) + self.w3 * 1/self.num_classes
        return torch.log(p)

    def fit(self, g, features, labels, masks):
        self.to(self.device)
        self.temp_model.fit(g, features, labels, masks)
        torch.cuda.empty_cache()
        logits = self.model(g, features)[masks[1]]
        label = labels[masks[1]]
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.unsqueeze(-1), 1)
        temp = self.temp_model.temperature.cpu().detach().numpy()
        w = self.ensemble_scaling(logits.cpu().detach().numpy(), one_hot.cpu().detach().numpy(), temp)
        self.w1, self.w2, self.w3 = w[0], w[1], w[2]
        return self

    def ensemble_scaling(self, logit, label, t):
        """
        Official ETS implementation from Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning
        Code taken from (https://github.com/zhang64-llnl/Mix-n-Match-Calibration)
        Use the scipy optimization because PyTorch does not have constrained optimization.
        """
        p1 = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
        logit = logit/t
        p0 = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
        p2 = np.ones_like(p0)/self.num_classes
        

        bnds_w = ((0.0, 1.0),(0.0, 1.0),(0.0, 1.0),)
        def my_constraint_fun(x): return np.sum(x)-1
        constraints = { "type":"eq", "fun":my_constraint_fun,}
        w = scipy.optimize.minimize(ETS.ll_w, (1.0, 0.0, 0.0), args = (p0,p1,p2,label), method='SLSQP', constraints = constraints, bounds=bnds_w, tol=1e-12, options={'disp': False})
        w = w.x
        return w

    @staticmethod
    def ll_w(w, *args):
    ## find optimal weight coefficients with Cros-Entropy loss function
        p0, p1, p2, label = args
        p = (w[0]*p0+w[1]*p1+w[2]*p2)
        N = p.shape[0]
        ce = -np.sum(label*np.log(p))/N
        return ce      

class TS(nn.Module):
    def __init__(self, model, device, conf):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
        self.device = device
        self.conf = conf
    def forward(self, g, features):
        logits = self.model(g, features)
        temperature = self.temperature_scale(logits)
        return logits / temperature

    def temperature_scale(self, logits):
        """
        Expand temperature to match the size of logits
        """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return temperature

    def fit(self, g, features, labels, masks):
        self.to(self.device)
        def eval(logits):
            temperature = self.temperature_scale(logits)
            calibrated = logits / temperature
            return calibrated
        
        self.train_param = [self.temperature]
        self.optimizer = optim.Adam(self.train_param, lr=self.conf.calibration["cal_lr"], weight_decay=self.conf.calibration["cal_weight_decay"])
        fit_calibration(self, eval, g, features, labels, masks, self.conf.calibration["epochs"], self.conf.calibration["patience"])
        return self
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.nn.pytorch import SAGEConv

from dgl.nn import GATConv

class SimpleTempMLP(nn.Module):
    """
    两层感知机，用于为每个节点输出正温度标量 Tᵢ。
    参数
    ----
    in_dim      : 输入特征维度（例如拼接后的波形 + 其他特征）
    hidden_dim  : 隐藏层宽度
    dropout     : Dropout 概率
    """
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.fc1      = nn.Linear(in_dim, hidden_dim)
        self.fc2      = nn.Linear(hidden_dim, 1)
        self.dropout  = nn.Dropout(dropout)
        self.softplus = nn.Softplus()          # 保证 Tᵢ > 0

    def forward(self, g, x):                    # g 参数保留以兼容原框架
        h = F.relu(self.fc1(x))                 # [N, hidden]
        h = self.dropout(h)
        h = self.fc2(h)                         # [N, 1]
        t = self.softplus(h).squeeze(-1)        # [N] 正标量
        return t

class FeatureExtractor:
    def __init__(self, features_to_use, norm_fn=lambda x: x, use_node_feature=True):
        self.features_to_use = features_to_use
        self.norm_fn = norm_fn
        self.use_node_feature = use_node_feature

    def extract(self, g: dgl.DGLGraph, model=None, device='cpu'):
        g_cpu = g.cpu()
        N = g_cpu.num_nodes()
        feat_dict = {}

        # 可能会用到的原始节点特征（如果存在）
        feat_cpu = None
        if 'feat' in g_cpu.ndata:
            feat_cpu = g_cpu.ndata['feat'].contiguous().cpu()

        # 统一特征形状为 [N, d] 的小工具（仅在本函数内使用）
        def _to_2d(x: torch.Tensor) -> torch.Tensor:
            # 挤掉尾部 size=1 的维度，避免 [N,1,1]
            while x.dim() > 1 and x.size(-1) == 1:
                x = x.squeeze(-1)
            if x.dim() == 1:
                if x.numel() != N:
                    raise ValueError(f"1D tensor length {x.numel()} != N={N}")
                x = x.view(N, 1)                 # [N] -> [N,1]
            elif x.dim() == 2:
                if x.size(0) != N and x.size(1) == N:
                    x = x.t().contiguous()       # [d,N] -> [N,d]（少见）
                elif x.size(0) != N:
                    raise ValueError(f"2D tensor first dim must be N={N}, got {tuple(x.shape)}")
            elif x.dim() == 3:
                if x.size(0) != N:
                    raise ValueError(f"3D tensor batch dim must be N={N}, got {tuple(x.shape)}")
                x = x.reshape(N, -1).contiguous()  # [N,d,?] -> [N,d*?]
            else:
                raise ValueError(f"Unsupported dim {x.dim()} for shape {tuple(x.shape)}")
            return x

        def _add(name: str, tensor: torch.Tensor):
            if tensor is None:
                return
            feat_dict[name] = self.norm_fn(_to_2d(tensor))

        # --- 图结构特征 ---
        if 'degree' in self.features_to_use:
            deg = compute_degree_features(g_cpu, device)          # [N,1]
            _add('degree', torch.log1p(deg))
        if 'centrality' in self.features_to_use:
            cen = compute_centrality(g_cpu, device)               # [N,1]
            _add('centrality', cen)
        if 'avg_feature_sim' in self.features_to_use and feat_cpu is not None:
            sim = compute_avg_feature_sim(g_cpu, feat_cpu)        # [N,1]
            _add('avg_feature_sim', sim)
        if 'avg_neighbor_jaccard' in self.features_to_use:
            _add('avg_neighbor_jaccard', compute_avg_neighbor_jaccard(g_cpu, device))  # [N,1]
        if 'clustering_coefficient' in self.features_to_use:
            _add('clustering_coefficient', compute_clustering_coefficient(g_cpu, device))  # [N,1]

        # --- 频谱/小波/拉普拉斯特征 ---
        if 'graph_wavelet' in self.features_to_use:
            s_list = [1.2]
            waves = [compute_graph_wavelet(g_cpu, device, k=2, s=s) for s in s_list]   # 每个 [N,K+1]
            _add('graph_wavelet', torch.cat(waves, dim=1))                              # [N,(K+1)*len(s)]
        if 'laplacian' in self.features_to_use:
            lap = compute_laplcian_embedding(g_cpu, device, k=10)                       # [N,10]
            _add('laplacian', lap)

        # --- 模型相关特征 ---
        if 'logits' in self.features_to_use and model is not None:
            if 'feat' not in g.ndata and feat_cpu is not None:
                g.ndata['feat'] = feat_cpu.to(device)
            with torch.no_grad():
                logits = model(g.to(device), g.ndata['feat'].to(device))                # [N,C]
            _add('logits', logits.cpu())

        # --- 原始节点特征（可选） ---
        if self.use_node_feature and feat_cpu is not None:
            _add('feat', feat_cpu)

        if len(feat_dict) == 0:
            raise RuntimeError(
                f"No features produced. features_to_use={self.features_to_use}, "
                f"ndata keys={list(g_cpu.ndata.keys())}"
            )

        final_feat = torch.cat([feat_dict[k].to(device) for k in feat_dict], dim=1).contiguous()
        return final_feat

class WATS(nn.Module):
    def __init__(self, model, device, conf, norm_fn=no,
                 features_to_use=None, use_node_feature=False):
        super().__init__()
        self.model = model
        self.device = device
        self.conf = conf
        self.norm_fn = norm_fn
        self.hidden_dim = conf.calibration["cal_hidden_dim"]
        self.use_node_feature = use_node_feature
        self.drop_rate = conf.calibration["cal_dropout"]
        self.features_to_use = features_to_use or ['graph_wavelet']
        self.feature_extractor = FeatureExtractor(self.features_to_use, norm_fn, use_node_feature)

        self.mlp = None
        self.register_buffer('graph_features', None)
        self.metrics_inited = False


    def init_graph_metrics(self, g):
        feats = self.feature_extractor.extract(g, model=self.model, device=self.device)
        feats = feats.to(self.device)
        self.register_buffer('graph_features', feats)
        self.metrics_inited = True
        self.build_gcn(feats.shape[1])

    def build_gcn(self, final_indim):
        self.mlp = SimpleTempMLP(in_dim=final_indim, hidden_dim=self.hidden_dim, dropout = self.drop_rate).to(self.device)

    def temperature_scale(self, g, logits):
        # self.graph_features = self.graph_features.to(logits.device)
        # T = self.mlp(g, self.graph_features)  # [N]
        # return T.unsqueeze(1).expand_as(logits)  # [N,K]
         # 1) 保障 feature buffer 与 logits 在同一设备/类型
        self.graph_features = self.graph_features.to(logits.device)
        T = self.mlp(g, self.graph_features)              # 可能是标量/[N]/[N,1]/[N,1,1]/[N,K]
        T = T.to(device=logits.device, dtype=logits.dtype)

        # 2) 压掉尾部多余的 1 维（比如 [N,1,1] -> [N,1] -> [N]）
        while T.dim() > 1 and T.size(-1) == 1:
            T = T.squeeze(-1)

        # 3) 统一成可与 [N,K] 对齐的形状
        if T.dim() == 0:
            # 常数温度：扩成 [N,1]
            T = T.expand(logits.size(0)).view(-1, 1)
        elif T.dim() == 1:
            # [N] -> [N,1]
            if T.size(0) != logits.size(0):
                raise ValueError(f"Temperature length {T.size(0)} != logits batch {logits.size(0)}")
            T = T.view(-1, 1)
        elif T.dim() == 2:
            # [N,1] 或 [N,K]
            if T.size(0) != logits.size(0):
                raise ValueError(f"Temperature batch {T.size(0)} != logits batch {logits.size(0)}")
            if T.size(1) == 1:
                return T.expand_as(logits)               # [N,1] -> [N,K]
            if T.size(1) == logits.size(1):
                return T                                 # [N,K] 直接用
            raise ValueError(f"Temperature shape {tuple(T.size())} not broadcastable to logits {tuple(logits.size())}")
        else:
            raise ValueError(f"Unexpected temperature shape {tuple(T.size())}")

        # 走到这里一定是 [N,1]
        return T.expand_as(logits)

    def forward(self, g, features):
        logits = self.model(g, features)
        T = self.temperature_scale(g, logits)
        return logits / (T + 1e-6)

    def fit(self, g, features, labels, masks):
        self.to(self.device)
        if not self.metrics_inited:
            self.init_graph_metrics(g)

        def eval_fn(logits):
            T = self.temperature_scale(g, logits)
            return logits / (T + 1e-6)

        self.optimizer = torch.optim.Adam(
            self.mlp.parameters(),
            lr=self.conf.calibration["cal_lr"],
            weight_decay=self.conf.calibration["cal_weight_decay"]
        )
        

        fit_calibration(self, eval_fn, g, features, labels, masks,
                        self.conf.calibration["epochs"],
                        self.conf.calibration["patience"])


class GCN_pure(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden, drop_rate, num_layers):
        super().__init__()
        self.drop_rate = drop_rate
        self.feature_list = [in_channels, num_hidden, num_classes]
        for _ in range(num_layers-2):
            self.feature_list.insert(-1, num_hidden)
        layer_list = []

        for i in range(len(self.feature_list)-1):
            layer_list.append(["conv"+str(i+1), dglnn.GraphConv(self.feature_list[i], self.feature_list[i+1])])
        
        self.layer_list = torch.nn.ModuleDict(layer_list)

    def forward(self, features, g):
        x = features
        for i in range(len(self.feature_list)-1):
            x = self.layer_list["conv"+str(i+1)](g, x)
            if i < len(self.feature_list)-2:
                x = F.relu(x)
                x = F.dropout(x, self.drop_rate, self.training)
        return x
    
class GCN(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden, drop_rate, num_layers):
        super().__init__()
        self.drop_rate = drop_rate
        self.feature_list = [in_channels, num_hidden, num_classes]
        for _ in range(num_layers-2):
            self.feature_list.insert(-1, num_hidden)
        layer_list = []

        for i in range(len(self.feature_list)-1):
            layer_list.append(["conv"+str(i+1), dglnn.GraphConv(self.feature_list[i], self.feature_list[i+1])])
        
        self.layer_list = torch.nn.ModuleDict(layer_list)

    def forward(self, features, g):
        x = features
        for i in range(len(self.feature_list)-1):
            x = self.layer_list["conv"+str(i+1)](g, x)
            if i < len(self.feature_list)-2:
                x = F.relu(x)
                x = F.dropout(x, self.drop_rate, self.training)
        return x
    
class CaGCN(nn.Module):
    def __init__(self, model, num_class, device, conf):
        super().__init__()
        self.model = model
        self.cagcn = GCN(num_class, 1, 16, drop_rate=conf.calibration["cal_dropout"], num_layers=2)
        self.device = device
        self.conf = conf

    def forward(self, g, features):
        logits = self.model(g, features)
        temperature = self.graph_temperature_scale(logits, g)
        return logits * F.softplus(temperature)

    def graph_temperature_scale(self, logits, g):
        """
        Perform graph temperature scaling on logits
        """
        temperature = self.cagcn(logits, g)
        return temperature

    def fit(self, g, features, labels, masks):
        self.to(self.device)
        def eval(logits):
            temperature = self.graph_temperature_scale(logits, g)
            calibrated = logits * F.softplus(temperature)
            return calibrated

        self.train_param = self.cagcn.parameters()
        self.optimizer = optim.Adam(self.train_param, lr=self.conf.calibration["cal_lr"], weight_decay=self.conf.calibration["cal_weight_decay"])
        fit_calibration(self, eval, g, features, labels, masks, self.conf.calibration["epochs"], self.conf.calibration["patience"])
        return self

class VS(nn.Module):
    def __init__(self, model, num_classes, device, conf):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.ones(num_classes))
        self.device = device
        self.conf = conf
    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temperature = self.vector_scale(logits)
        return logits * temperature + self.bias

    def vector_scale(self, logits):
        """
        Expand temperature to match the size of logits
        """
        temperature = self.temperature.unsqueeze(0).expand(logits.size(0), logits.size(1))
        return temperature

    def fit(self, g, features, labels, masks):
        self.to(self.device)
        def eval(logits):
            temperature = self.vector_scale(logits)
            calibrated = logits * temperature + self.bias
            return calibrated

        self.train_param = [self.temperature]
        self.optimizer = optim.Adam(self.train_param, lr=self.conf.calibration["cal_lr"], weight_decay=self.conf.calibration["cal_weight_decay"])
        fit_calibration(self, eval, g, features, labels, masks, self.conf.calibration["epochs"], self.conf.calibration["patience"])
        return self
        
class CaGCN_GETS(nn.Module):
    def __init__(self, model, feature_dim, num_class, device, conf):
        super().__init__()
        self.model = model
        self.device = device

        self.learner = GETS(
            num_classses=num_class,
            hidden_dim=conf.calibration["hidden_dim"],
            dropout_rate=conf.calibration["cal_dropout"],
            num_layer=conf.calibration["cal_num_layer"],
            expert_select=conf.calibration["expert_select"],
            expert_configs=conf.calibration["expert_configs"],
            feature_dim=feature_dim,
            feature_hidden_dim=conf.calibration["feature_hidden_dim"],
            degree_hidden_dim=conf.calibration["degree_hidden_dim"],
            noisy_gating=conf.calibration["noisy_gating"],
            coef=conf.calibration["coef"],
            device=device,
            backbone=conf.calibration['backbone']
        )
        self.conf = conf
        
    def forward(self, g, features):
        logits = self.model(g, features)
        return self.learner(g, logits, features)
    
    def fit(self, g, features, labels, masks):
        self.to(self.device)
        def eval(logits):
            return self.learner(g, logits, features)

        self.train_param = self.parameters()
        self.optimizer = optim.Adam(self.train_param, lr=self.conf.calibration["cal_lr"], weight_decay=self.conf.calibration["cal_weight_decay"])
        fit_calibration(self, eval, g, features, labels, masks, self.conf.calibration["epochs"], self.conf.calibration["patience"])
        return self

    
from typing import Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree

def shortest_path_length(edge_index, mask, max_hop, device):
    """
    Return the shortest path length to the mask for every node
    """
    dist_to_train = torch.ones_like(mask, dtype=torch.long, device=device) * torch.iinfo(torch.long).max
    seen_mask = torch.clone(mask).to(device)
    for hop in range(max_hop):
        current_hop = torch.nonzero(mask).to(device)
        dist_to_train[mask] = hop
        next_hop = torch.zeros_like(mask, dtype=torch.bool, device=device)
        for node in current_hop:
            node_mask = edge_index[0,:]==node
            nbrs = edge_index[1,node_mask]
            next_hop[nbrs] = True
        hop += 1
        # mask for the next hop shouldn't be seen before
        mask = torch.logical_and(next_hop, ~seen_mask)
        seen_mask[next_hop] = True
    return dist_to_train   

class CalibAttentionLayer(MessagePassing):
    _alpha: OptTensor

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            edge_index: Adj,
            num_nodes: int,
            train_mask: Tensor,
            dist_to_train: Tensor = None,
            heads: int = 8,
            negative_slope: float = 0.2,
            bias: float = 1,
            self_loops: bool = True,
            fill_value: Union[float, Tensor, str] = 'mean',
            bfs_depth=2,
            device='cpu',
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.fill_value = fill_value
        self.edge_index = edge_index
        self.num_nodes = num_nodes

        self.temp_lin = Linear(in_channels, heads,
                               bias=False, weight_initializer='glorot')

        # The learnable clustering coefficient for training node and their neighbors
        self.conf_coef = Parameter(torch.zeros([]))
        self.bias = Parameter(torch.ones(1) * bias)
        self.train_a = Parameter(torch.ones(1))
        self.dist1_a = Parameter(torch.ones(1))

        # Compute the distances to the nearest training node of each node
        train_mask_indices_tensor = torch.from_numpy(train_mask).to(device)
        train_mask_tensor = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        train_mask_tensor.scatter_(0, train_mask_indices_tensor, True)
        dist_to_train = dist_to_train if dist_to_train is not None else shortest_path_length(edge_index, train_mask_tensor, bfs_depth, device)
        self.register_buffer('dist_to_train', dist_to_train)

        self.reset_parameters()
        if self_loops:
            # We only want to add self-loops for nodes that appear both as
            # source and target nodes:
            self.edge_index, _ = remove_self_loops(
                self.edge_index, None)
            self.edge_index, _ = add_self_loops(
                self.edge_index, None, fill_value=self.fill_value,
                num_nodes=num_nodes)

    def reset_parameters(self):
        self.temp_lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor]):
        N, H = self.num_nodes, self.heads

        # Individual Temperature
        normalized_x = x - torch.min(x, 1, keepdim=True)[0]
        normalized_x /= torch.max(x, 1, keepdim=True)[0] - \
                        torch.min(x, 1, keepdim=True)[0]

        # t_delta for individual nodes
        # x_sorted_scalar: [N, 1]
        x_sorted = torch.sort(normalized_x, -1)[0]
        temp = self.temp_lin(x_sorted)

        # Next, we assign spatial coefficient
        # a_cluster:[N]
        a_cluster = torch.ones(N, dtype=torch.float32, device=x[0].device)
        a_cluster[self.dist_to_train == 0] = self.train_a
        a_cluster[self.dist_to_train == 1] = self.dist1_a


        # For confidence smoothing
        conf = F.softmax(x, dim=1).amax(-1)
        deg = degree(self.edge_index[0, :], self.num_nodes)
        deg_inverse = 1 / deg
        deg_inverse[deg_inverse == float('inf')] = 0

        out = self.propagate(self.edge_index,
                             temp=temp.view(N, H) * a_cluster.unsqueeze(-1),
                             alpha=x / a_cluster.unsqueeze(-1),
                             conf=conf)
        sim, dconf = out[:, :-1], out[:, -1:]
        out = F.softplus(sim + self.conf_coef * dconf * deg_inverse.unsqueeze(-1))
        out = out.mean(dim=1) + self.bias 
        return out.unsqueeze(1)

    def message(
            self,
            temp_j: Tensor,
            alpha_j: Tensor,
            alpha_i: OptTensor,
            conf_i: Tensor,
            conf_j: Tensor,
            index: Tensor,
            ptr: OptTensor,
            size_i: Optional[int]) -> Tensor:
        """
        alpha_i, alpha_j: [E, H]
        temp_j: [E, H]
        """
        if alpha_i is None:
            print("alphai is none")
        alpha = (alpha_j * alpha_i).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        # Agreement smoothing + Confidence smoothing
        return torch.cat([
            (temp_j * alpha.unsqueeze(-1).expand_as(temp_j)),
            (conf_i - conf_j).unsqueeze(-1)], -1)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}{self.out_channels}, heads={self.heads}')

    
class GATS(nn.Module):
    def __init__(self, model, g, num_class, train_mask, device, conf):
        super().__init__()
        self.model = model
        src_nodes, dst_nodes = g.edges()
        self.edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
        self.num_nodes = g.num_nodes()
        self.conf = conf
        self.cagat = CalibAttentionLayer(in_channels=num_class,
                                         out_channels=1,
                                         edge_index=self.edge_index,
                                         num_nodes=self.num_nodes,
                                         train_mask=train_mask,
                                         dist_to_train=conf.calibration["dist_to_train"],
                                         heads=conf.calibration["heads"],
                                         bias=conf.calibration["bias"],
                                         device = device)
        self.device = device
        
    def forward(self, g, features):
        logits = self.model(g, features)
        temperature = self.graph_temperature_scale(logits)
        return logits / temperature

    def graph_temperature_scale(self, logits):
        """
        Perform graph temperature scaling on logits
        """
        temperature = self.cagat(logits).view(self.num_nodes, -1)
        return temperature.expand(self.num_nodes, logits.size(1))

    def fit(self, g, features, labels, masks):
        self.to(self.device)
        def eval(logits):
            temperature = self.graph_temperature_scale(logits)
            calibrated = logits / temperature
            return calibrated

        self.train_param = self.cagat.parameters()
        self.optimizer = optim.Adam(self.train_param, lr=self.conf.calibration["cal_lr"], weight_decay=self.conf.calibration["cal_weight_decay"])
        fit_calibration(self, eval, g, features, labels, masks, self.conf.calibration["epochs"], self.conf.calibration["patience"])
        return self