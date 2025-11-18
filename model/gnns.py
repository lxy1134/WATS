import dgl.nn as dglnn
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl


def load_gnn(conf):
    if conf.gnn["type"] == "gcn":
        return GCN(
            conf.gnn["in_dim"],
            conf.gnn["hid_dim"],
            conf.gnn["out_dim"],
            conf.gnn["num_layer"],
            conf.gnn["dropout"],
            conf.gnn["norm"]
        )
        
    elif conf.gnn["type"] == "gat":
        return GAT(
            conf.gnn["in_dim"],
            conf.gnn["hid_dim"],
            conf.gnn["out_dim"],
            conf.gnn["num_layer"],
            conf.gnn["dropout"],
            conf.gnn["norm"]
        )
    elif conf.gnn["type"] == "gin":
        return GIN(
            conf.gnn["in_dim"],
            conf.gnn["hid_dim"],
            conf.gnn["out_dim"],
            conf.gnn["num_layer"],
            conf.gnn["dropout"],
            conf.gnn["norm"]
        )
    elif conf.gnn["type"] == "gcnii":
        return GCNII(
            conf.gnn["in_dim"],
            conf.gnn["hid_dim"],
            conf.gnn["out_dim"],
            conf.gnn["num_layer"], # 建议在配置文件中将此值设大，如 8, 16
            conf.gnn["dropout"],
            alpha=conf.gnn.get("alpha", 0.1), # 从配置读取或使用默认值
            lamda=conf.gnn.get("lamda", 0.5), # 从配置读取或使用默认值
            norm=conf.gnn["norm"]
        )
    else:
        raise NotImplementedError

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer, dropout, norm):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = norm
        if norm:
            self.norms = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size)
        )
        if norm:
            self.norms.append(nn.BatchNorm1d(hid_size))
        for _ in range(num_layer - 2):
            self.layers.append(
                dglnn.GraphConv(hid_size, hid_size)
            )
            if norm:
                self.norms.append(nn.BatchNorm1d(hid_size))
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.layers) - 1:
                if self.norm:
                    h = self.norms[i](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h
    
class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer, dropout, norm):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = norm
        if norm:
            self.norms = nn.ModuleList()
        num_heads = 2 
        
        # First GAT layer
        self.layers.append(
            dglnn.GATConv(in_size, hid_size, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout)
        )
        if norm:
            self.norms.append(nn.BatchNorm1d(hid_size * num_heads))
        
        # # Middle GAT layers
        # for _ in range(num_layer - 2):
        #     self.layers.append(
        #         dglnn.GATConv(hid_size * num_heads, hid_size, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout)
        #     )
        #     if norm:
        #         self.norms.append(nn.BatchNorm1d(hid_size * num_heads))
        
        # Output GAT layer
        self.final_project = nn.Linear(hid_size* num_heads , out_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.layers) - 1:
                if self.norm:
                    h = self.norms[i](h)
                h = F.relu(h)
                h = self.dropout(h)
        h = self.final_project(h.view(h.size(0), -1))
        return h
    
class GIN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer, dropout, norm):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = norm
        if norm:
            self.norms = nn.ModuleList()
        # two-layer GIN
        self.layers.append(
            dglnn.GINConv(
                nn.Sequential(
                    nn.Linear(in_size, hid_size)
                ),
                'mean'
            )
        )
        if norm:
            self.norms.append(nn.BatchNorm1d(hid_size))
        for _ in range(num_layer - 2):
            self.layers.append(
                dglnn.GINConv(
                    nn.Sequential(
                        nn.Linear(hid_size, hid_size)
                    ),
                    'mean'
                )
            )
            if norm:
                self.norms.append(nn.BatchNorm1d(hid_size))
        self.layers.append(
            dglnn.GINConv(
                nn.Sequential(
                    nn.Linear(hid_size, out_size)
                ),
                'mean'
            )
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.layers) - 1:
                if self.norm:
                    h = self.norms[i](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h


# ... (在 gnns.py 现有代码之后添加)

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        with g.local_scope():
            # DGL 消息传递实现标准的 GCN 聚合
            # h = D^-0.5 * A * D^-0.5 * h * W
            # 这里我们先进行线性变换，再进行图聚合，或者反过来，取决于具体公式
            # 为了匹配 GCNII 的特定公式 (1-beta)I + beta L，通常 GCNII 层定义略有不同
            # 但在 DGL 中，我们可以使用 dglnn.GraphConv 作为基础，或者手动实现
            # 为了简化并确保正确性，我们使用 DGL 的 GraphConv 并配合外部的 APPNP 风格或手动残差
            pass
        # 注意：GCNII 的核心层逻辑比较特殊，建议直接使用 DGL 官方或标准 PyTorch 实现。
        # 下面提供一个完整的、不依赖 DGL 内置复杂层的 GCNIILayer 实现。
        pass

# 更推荐：直接使用 dglnn 现有的组件组合，或者如下的标准 GCNII 实现

class GCNIILayer(nn.Module):
    def __init__(self, n_channels, alpha=0.1, beta=0.1, weight_decay=0.0, activation=None):
        super(GCNIILayer, self).__init__()
        self.n_channels = n_channels
        self.alpha = alpha
        self.beta = beta
        self.weight_decay = weight_decay
        self.activation = activation
        self.linear = nn.Linear(n_channels, n_channels)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.n_channels)
        self.linear.weight.data.uniform_(-stdv, stdv)
        self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h, h_0):
        # GCNII 公式: H^(l+1) = sigma( ((1-alpha)P H^(l) + alpha H^(0)) ((1-beta)I + beta W) )
        
        # 1. 图聚合 P * H^(l)
        with g.local_scope():
            # 假设 g 已经添加了自环并计算了归一化权重，或者使用 DGL 的 conv
            # 这里使用 DGL 的 GraphConv 的聚合逻辑：
            # 使用 dgl.nn.GraphConv(..., norm='both', weight=False, bias=False) 可以实现 P*H
            # 但为了简单，我们手动写消息传递：
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(h.device).unsqueeze(1)
            
            h_src = h * norm
            g.ndata['h'] = h_src
            g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'h'))
            ah = g.ndata.pop('h') * norm
            
            # 2. 初始残差 (Initial Residual): (1-alpha)AH + alpha H0
            h_supp = (1 - self.alpha) * ah + self.alpha * h_0
            
            # 3. 恒等映射 (Identity Mapping): ((1-beta)I + beta W)
            # h_final = h_supp @ ((1-beta)I + beta W) = (1-beta)h_supp + beta * (h_supp @ W)
            h_final = (1 - self.beta) * h_supp + self.beta * self.linear(h_supp)
            
            if self.activation:
                h_final = self.activation(h_final)
            
            return h_final

class GCNII(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer, dropout, 
                 alpha=0.1, lamda=0.5, norm=True):
        super(GCNII, self).__init__()
        self.fc_in = nn.Linear(in_size, hid_size)
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.act = F.relu
        self.norm = norm
        
        # GCNII 的层数通常比较深，例如 8, 16, 32
        for i in range(num_layer):
            # beta 通常随着层数增加而减小: log(lambda/l + 1)
            beta = math.log(lamda / (i + 1) + 1)
            self.layers.append(GCNIILayer(hid_size, alpha=alpha, beta=beta, activation=self.act))
        
        self.fc_out = nn.Linear(hid_size, out_size)
        
        if norm:
            self.norms = nn.ModuleList([nn.BatchNorm1d(hid_size) for _ in range(num_layer)])

    def forward(self, g, features):
        h = features
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.fc_in(h)
        h = F.relu(h)
        
        h_0 = h # 保存初始特征用于残差连接
        
        for i, layer in enumerate(self.layers):
            h = F.dropout(h, self.dropout, training=self.training)
            h = layer(g, h, h_0)
            if self.norm:
                h = self.norms[i](h)
        
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.fc_out(h)
        return h