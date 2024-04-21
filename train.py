from util import *
from preprocess import preprocess_defaults, preprocess_linear
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class SubsetDataset(torch.utils.data.Dataset):
    data_keys = ['agents', 'indices', 'costs', 'occupancy_iteration_starts', 'occupancy_iteration_lens', 'occupancy_indices', 'shortest_paths_indices']

    def __init__(self, data, a, is_train=True):
        super().__init__()
        self.a = a
        self.augment = self.is_train = is_train and (a.rotation_symmetry or a.reflection_symmetry)
        self.data = data
        self.arange_T = np.arange(a.T_data, dtype=np.uint64).reshape(1, -1)
        self.num_sampled_subsets = a.num_sampled_subsets and (a.num_sampled_subsets if is_train else a.num_subsets)
        if a.prepool > 1:
            idxs = data['occupancy_indices']
            row, col = idxs // a.G_orig.cols, idxs % a.G_orig.cols
            data['occupancy_indices'] = (row // a.prepool) * a.G.cols + (col // a.prepool)

            idxs = data['shortest_paths_indices']
            mask = idxs == np.iinfo(np.uint16).max
            row, col = idxs // a.G_orig.cols, idxs % a.G_orig.cols
            data['shortest_paths_indices'] = (row // a.prepool) * a.G.cols + (col // a.prepool)
            data['shortest_paths_indices'][mask] = np.iinfo(np.uint16).max

    def __len__(self):
        return int(len(self.data['indices']) // (self.a.num_subsets if self.num_sampled_subsets else 1) // a.reduce_data_factor)

    def process_labels(self, costs):
        a = self.a
        ret = Namespace(final=costs[..., 0], shortest=costs[..., 1], init=costs[..., 2])
        if a.label_discretization:
            delta = ret.final - ret.init # (*dims, S)
            percentiles = np.percentile(delta, a.label_discretization, axis=-1, keepdims=True) # (num_percentiles, *dims, 1)
            # new discretized labels, lower is better
            ret.label = (delta > percentiles).sum(axis=0)
        return ret

    def get_occupancy_idxs_2d(self, starts, lens, dT=1):
        mask = lens.reshape(-1, 1) > self.arange_T[:, ::dT]
        occ_idxs_idxs = (self.arange_T[:, ::dT] + starts.reshape(-1, 1))[mask]
        return self.data['occupancy_indices'][occ_idxs_idxs].astype(np.uint32), mask

    def get_occupancy_idxs_3d(self, starts, lens, dT=1):
        locations, mask = self.get_occupancy_idxs_2d(starts, lens, dT=dT)
        times = np.broadcast_to(self.arange_T[:, ::dT] // dT, mask.shape)[mask]
        return locations * ((self.a.T_data - 1) // dT + 1) + times

    def __getitem__(self, idx):
        a, data, G = self.a, self.data, self.a.G
        # 4 possible channels
        #   - shortest paths of subset drives
        #   - current paths
        #   - occupancy of other drives (nonstatic)
        #   - static obstacles
        use_init = 'init' in a.features
        use_shortest = 'shortest' in a.features
        use_separate_static = 'separate_static' in a.features

        T = (a.T_data - 1) // a.dT + 1
        sampled_subsets = self.num_sampled_subsets or 1
        occupancies = np.zeros((sampled_subsets, use_init + use_shortest + 1 + use_separate_static, G.rows, G.cols, T), dtype=np.float32)

        channel_offset = G.size * T
        if use_shortest: shortest_channel = 0
        if use_init: init_channel = use_shortest * channel_offset
        nonstatic_channel = (use_shortest + use_init) * channel_offset
        obstacles = np.zeros((G.size, T), dtype=np.float32)
        np.add.at(obstacles, a.obstacles, 1)
        obstacles = obstacles.reshape((G.rows, G.cols, T))

        idxs = (idx * a.num_subsets + np.random.choice(a.num_subsets, size=self.num_sampled_subsets, replace=False)) if self.num_sampled_subsets else [idx]
        for i, idx in enumerate(idxs):
            occupancy = occupancies[i]
            window, iteration = data['indices'][idx]
            subagents = data['agents'][idx]

            if use_shortest:
                # Shape (num_subset_agents, T)
                shortest_paths_indices = data['shortest_paths_indices'][window, subagents, :a.T_data :a.dT]
                shortest_paths_indices_3d = shortest_paths_indices.astype(np.uint32) * T + self.arange_T[:, ::a.dT] // a.dT
                shortest_paths_mask = shortest_paths_indices < np.iinfo(np.uint16).max
                # Similar to occupancy[shortest_paths_indices_3d] += 1, but will account for repeated indices
                np.add.at(occupancy.reshape(-1), shortest_channel + shortest_paths_indices_3d.flatten()[shortest_paths_mask.flatten()], 1)

            # Set occupancies for all agents to be 1
            iter_starts = data['occupancy_iteration_starts'][iteration]
            iter_lens = data['occupancy_iteration_lens'][iteration]
            iter_occ_idxs = self.get_occupancy_idxs_3d(iter_starts, iter_lens, dT=a.dT)
            # Similar to occupancy.reshape(-1)[nonstatic_channel + iter_occ_idxs] = 1, but will account for repeated indices which occurs during pooling
            np.add.at(occupancy.reshape(-1), nonstatic_channel + iter_occ_idxs, 1)

            sub_occ_idxs = self.get_occupancy_idxs_3d(iter_starts[subagents], iter_lens[subagents], dT=a.dT)
            # Set occupancies for subset agents at the nonstatic obstacle channel to be 0
            # Similar to occupancy.reshape(-1)[nonstatic_channel + sub_occ_idxs] = 0
            np.add.at(occupancy.reshape(-1), nonstatic_channel + sub_occ_idxs, -1)

            # Set occupancies for subset agents at the init channel to be 1
            if use_init:
                # Similar to occupancy.reshape(-1)[init_channel + sub_occ_idxs] = 1
                np.add.at(occupancy.reshape(-1), init_channel + sub_occ_idxs, 1)

            # Set obstacle occupancies
            # Similar to occupancy[-1, a.obstacle_rows, a.obstacle_cols] = 1
            # np.add.at(occupancy.reshape(-1, T), obstacle_channel + a.obstacles, 1)

            if self.augment:
                aug_vert, aug_hor = torch.rand(2).numpy() < 0.5
                if not a.reflection_symmetry: aug_hor = aug_vert # There should only be rotational symmetry, not reflective unless the directed graph is fed in as input
                if aug_vert: occupancy = occupancy[:, ::-1]
                if aug_hor: occupancy = occupancy[:, :, ::-1]
                occupancy = np.ascontiguousarray(occupancy)
        
        occupancies[:, -1] += obstacles

        return Namespace(occupancy=occupancies if self.num_sampled_subsets else occupancies[0], costs=self.process_labels(data['costs'][idxs if self.num_sampled_subsets else idx].astype(np.float32)))

class MultiSubsetDataset(SubsetDataset):
    def __len__(self):
        return int(len(self.data['indices']) // self.a.num_subsets // a.reduce_data_factor)

    def __getitem__(self, idx):
        a, data, G = self.a, self.data, self.a.G
        use_shortest = 'shortest' in a.features

        aug_vert, aug_hor = torch.rand(2).numpy() < 0.5 if self.augment else (False, False)
        if not a.reflection_symmetry: aug_hor = aug_vert # There should only be rotational symmetry, not reflective
        def augment(locations):
            row, col = locations // G.cols, locations % G.cols
            if aug_vert: row = G.rows - row - 1
            if aug_hor: col = G.cols - col - 1
            return row * G.cols + col

        i_start = idx * a.num_subsets
        window, iteration = data['indices'][i_start]

        ret = Namespace(
            subagents=data['agents'][i_start: i_start + a.num_subsets].astype(int),
            costs=self.process_labels(data['costs'][i_start: i_start + a.num_subsets].astype(np.float32)),
        )

        if use_shortest:
            shortest_paths_locations_ = data['shortest_paths_indices'][window]
            shortest_paths_mask_ = shortest_paths_locations_ < np.iinfo(np.uint16).max
            shortest_paths_lengths = shortest_paths_mask_.sum(axis=-1)
            shortest_paths_locations_ = shortest_paths_locations_[:, :a.T_data].astype(int) # (num_agents, a.T_data)
            ret.shortest = Namespace(locations_=augment(shortest_paths_locations_), lengths=shortest_paths_lengths)

        iter_starts = data['occupancy_iteration_starts'][iteration]
        init_paths_lengths = data['occupancy_iteration_lens'][iteration].astype(int)
        init_locations, init_mask_ = self.get_occupancy_idxs_2d(iter_starts, init_paths_lengths)
        init_locations_ = np.zeros_like(init_mask_, dtype=int)
        init_locations_[init_mask_] = init_locations
        ret.init = Namespace(locations_=augment(init_locations_), lengths=init_paths_lengths)
        return ret

class Network(nn.Module):
    def __init__(self, a, train_data=None):
        super().__init__()
        self.a = a

        if a.loss == 'pairwise_svm':
            self.bias = nn.Parameter(torch.tensor(0, dtype=torch.float32))

        label_shift = 0
        if train_data is not None:
            costs = train_data['costs']
            final, shortest, init = costs[..., 0], costs[..., 1], costs[..., 2]
            label = final - {None: 0, 'init': init, 'shortest': shortest}[a.label_shift]
            label_shift = label[~(np.isinf(label) | np.isnan(label))].mean()
        self.register_buffer('label_shift', torch.tensor(label_shift, dtype=torch.float32))

    def out(self, preds, costs):
        a = self.a
        out = Namespace()
        if a.loss == 'mse':
            shortest, init = (costs.get(x, None) for x in ['shortest',  'init'])
            preds = preds + {None: 0, 'init': init, 'shortest': shortest}[a.label_shift] + self.label_shift
            preds = preds + (shortest - preds).detach().clamp(min=0)
            preds = preds - init
        out.preds = preds
        if costs.get('final') is not None:
            out.loss = self.loss(out, costs)
        return out

    def loss(self, out, costs):
        a = self.a
        preds = out.preds
        final, init = costs.final, costs.init
        if a.loss == 'mse':
            target = costs.label if a.label_discretization else (final - init)
            if a.mse_clip:
                preds = preds - preds.detach().clamp(min=0)
                target = target.clamp(max=0)
            loss = F.huber_loss(preds, target, delta=a.huber_delta) if a.huber_delta else F.mse_loss(preds, target)
        elif a.loss == 'pairwise_svm':
            *dims, S = preds.shape
            target = costs.label if a.label_discretization else (final - init)
            if a.mse_clip:
                target = target.clamp(max=0)
            pair_target = (target.view(*dims, S, 1) - target.view(*dims, 1, S)).sign()
            pair_preds = preds.view(*dims, S, 1) - preds.view(*dims, 1, S) + self.bias
            loss = (pair_target.abs() - pair_target * pair_preds).clamp(min=0).mean()
        return loss

    def get_opt(self, a):
        if a.weight_decay == 0: return optim.Adam(self.parameters(), lr=a.lr)
        decay, no_decay = set(), set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        assert len(decay & no_decay) == 0
        assert len(param_dict.keys() - (decay | no_decay)) == 0
        return torch.optim.Adam([
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": a.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ], lr=a.lr)

class SubsetNetwork(Network):
    def __init__(self, a, train_data=None, in_channels=None):
        super().__init__(a, train_data)
        # Input shape (batch, H=77, W=37, T=34)
        use_init = 'init' in a.features
        use_shortest = 'shortest' in a.features
        use_separate_static = 'separate_static' in a.features
        assert use_init or use_shortest
        C = in_channels or (1 + use_init + use_shortest + use_separate_static)
        depth_1 = 32
        padding_0 = a.padding_0
        pool3d_stride_0 = 3
        pool3d_stride_1 = 2
        if a.map in ['random-32-32-10.map', 'empty-32-32.map', 'maze-32-32-4.map']:
            pool3d_pad_0 = 0
            pool3d_pad_1 = 0
            pool2d_stride_0 = 1
        elif a.map == 'den312d.map':
            depth_1 = 16
            pool3d_pad_0 = 0
            pool3d_stride_1 = 3
            pool3d_pad_1 = 0
            pool2d_stride_0 = 1
        elif a.map.startswith('warehouse'):
            if a.prepool == 1:
                pool3d_pad_0 = 0
                pool3d_stride_1 = 3
                pool3d_pad_1 = 0
                pool2d_stride_0 = [2, 1]
            elif a.prepool == 2:
                pool3d_pad_0 = 0
                pool3d_stride_0 = [2, 3, 3]
                pool3d_stride_1 = 2
                pool3d_pad_1 = 0
                pool2d_stride_0 = [1, 2]
        elif a.map == 'ost003d.map':
            assert (a.prepool, a.dT) == (4, 6)
            pool3d_pad_0 = 0
            pool3d_stride_0 = [2, 2, 3]
            pool3d_stride_1 = 2
            pool3d_pad_1 = 0
            pool2d_stride_0 = 1
        elif a.map == 'den520d.map':
            assert (a.prepool, a.dT) == (4, 6)
            pool3d_pad_0 = 0
            pool3d_stride_0 = [3, 3, 3]
            pool3d_stride_1 = 2
            pool3d_pad_1 = 0
            pool2d_stride_0 = 1

        self.location_embedding = nn.Embedding(a.G.size, C) if a.location_embedding else None
        ev_globals, ev_locals = globals(), locals()
        ev = lambda layer: eval('nn.' + layer, ev_globals, ev_locals)
        self.occupancy_cnn = nn.Sequential(*([
            nn.Conv3d(C, depth_1, a.kernel_size_0, padding=padding_0),
            nn.BatchNorm3d(depth_1, momentum=0.2),
            nn.ReLU(),
            nn.MaxPool3d(pool3d_stride_0, padding=pool3d_pad_0),
            nn.Conv3d(depth_1, 64, 3),
            nn.BatchNorm3d(64, momentum=0.2),
            nn.ReLU(),
            nn.MaxPool3d(pool3d_stride_1, padding=pool3d_pad_1)
        ] if a.subset_convwht == 'default' else map(ev, a.subset_convwht)))
        print(f'Input shape {(C, a.G.rows, a.G.cols, (a.T - 1) // a.dT + 1)}')
        C, H, W, T = compute_output_shape((C, a.G.rows, a.G.cols, (a.T - 1) // a.dT + 1), self.occupancy_cnn) # e.g. (64, 12, 6, 4)
        print(f'Intermediate tensor shape ({C}, {H}, {W}, {T}) after 3D convolutions')

        if a.subset_convwh is not None:
            self.time_fc = nn.Linear(T, 1) if T > 1 else lambda x: x
            eval_globals, eval_locals = globals(), locals()
            self.merged_cnn = nn.Sequential(*([
                nn.Conv2d(C, 128, 3, padding=1),
                nn.BatchNorm2d(128, momentum=0.2),
                nn.ReLU(),
                nn.MaxPool2d(pool2d_stride_0),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64, momentum=0.2),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ] if a.subset_convwh == 'default' else map(ev, a.subset_convwht)))
            (C, H, W), T = compute_output_shape((C, H, W), self.merged_cnn), 1 # e.g. (64, 3, 3)
            print(f'Intermediate tensor shape ({C}, {H}, {W}) after 2D convolutions')

        in_features = C * H * W * T
        self.out_fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        ) if a.subset_fc == 'default' else nn.Sequential(*(eval('nn.' + layer) for layer in a.subset_fc))

    def forward(self, kwargs):
        a = self.a
        occupancy, costs = kwargs.occupancy, kwargs.costs
        B, *num_sampled_subsets, C, H, W, T = occupancy.shape
        occupancy = occupancy.view(-1, C, H, W, T)
        if self.location_embedding:
            occupancy = occupancy + self.location_embedding.weight.T.view(1, C, H, W, 1).expand_as(occupancy)
        x = self.occupancy_cnn(occupancy)  # shape (n_batch, depth=16, w=12, h=6, t=4)
        if a.subset_convwh is not None:
            x = self.merged_cnn(self.time_fc(x).squeeze(dim=-1))
        x = self.out_fc(x.view(B, *num_sampled_subsets, -1)).squeeze(-1)
        return self.out(x, costs)

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = self.kdim = self.vdim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
        self.q, self.k, self.v = None, None, None

    def forward(self, x, attn_mask=None):
        B, T, embed_dim = x.shape
        H, head_dim = self.num_heads, self.head_dim
        q, k, v = [y.view(B, T, H, head_dim) for y in F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)]
        attn = torch.einsum('bthd,bThd->bhtT', q, k) / math.sqrt(head_dim)
        if attn_mask is not None:
            attn += attn_mask.view(B, 1, 1, T)
        attn = attn.softmax(dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        out = torch.einsum('bhtT,bThd->bthd', attn, v)
        return self.out_proj(out.reshape(B, T, embed_dim))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, feedforward_factor, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)

        self.dim_feedforward = feedforward_factor * d_model
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        if self.dim_feedforward > 0:
            self.linear1 = nn.Linear(d_model, self.dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(self.dim_feedforward, d_model)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.dropout2 = nn.Dropout(dropout)
            self.activation = activation

    def forward(self, x, attn_mask=None):
        return self.feedforward_block(self.self_attn_block(x, attn_mask=attn_mask))
    
    def self_attn_block(self, x, attn_mask=None):
        return self.norm1(x + self.dropout1(self.self_attn(x, attn_mask=attn_mask)))
    
    def feedforward_block(self, x):
        return self.norm2(x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))) if self.dim_feedforward > 0 else x

class TransformerEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MultiSubsetNetwork(Network):
    def __init__(self, a, train_data=None):
        super().__init__(a, train_data)
        assert a.loss != 'mse' or a.mse_clip

        self.embed_dim = a.hidden_dims[0]
        if 'shortest' in a.features:
            self.embed_dim //= 2
            self.shortest_embedding = nn.Embedding((a.T - 1) // a.dT + 1, self.embed_dim)
        self.init_embedding = nn.Embedding((a.T - 1) // a.dT + 1, self.embed_dim)
        self.obstacle_embedding = nn.Embedding(1, a.hidden_dims[0])
        self.location_embedding = nn.Embedding(a.G.size, a.hidden_dims[0])

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv3d(dim1, dim2, 3, padding=1),
            *lif(a.use_batch_norm, nn.BatchNorm3d(dim2, momentum=0.2)),
            nn.ReLU(),
        ) for dim1, dim2 in zip(a.hidden_dims, a.hidden_dims[1:])]) if a.use_3d_convolution else [None for dim in a.hidden_dims[1:]]

        self.attns = nn.ModuleList([
            TransformerEncoderLayer(dim, a.num_heads, a.ff_factor)
        for dim in a.hidden_dims[1:]]) if a.use_path_attention else [None for dim in a.hidden_dims[1:]]

        sub_hidden_dim = a.hidden_dims[-1]
        self.sub_attn_embedding = nn.Embedding(1, sub_hidden_dim)
        self.sub_attns = TransformerEncoder([TransformerEncoderLayer(sub_hidden_dim, a.num_heads, a.ff_factor_sub) for _ in range(a.num_subset_attention_layers)])

        out_dims = [sub_hidden_dim, *a.out_dims]
        self.out_fc = nn.Sequential(
            *[layer for dim1, dim2 in zip(out_dims, out_dims[1:]) for layer in (nn.Linear(dim1, dim2), nn.ReLU())],
            nn.Linear(out_dims[-1], 1),
        )

    def forward(self, kwargs):
        a, G = self.a, self.a.G
        init, shortest, subagents, costs = Namespace(**kwargs.init), Namespace(**kwargs.get('shortest', {})), kwargs.subagents, kwargs.costs

        B, A = init.lengths.shape
        B, S, SA = subagents.shape # _, num_subsets, num_subset_agents
        device = init.lengths.device
        T_ = a.T // a.dT
        arange_T_ = torch.arange(T_, device=device).view(1, 1, T_)
        arange_B = torch.arange(B, device=device).view(B, 1, 1)
        times_ = arange_T_.expand(B, A, T_)
        batch_ = arange_B.expand(B, A, T_)

        for d in [init, *lif(shortest, shortest)]:
            if d.locations_.size(2) != T_:
                d.lengths = (d.lengths.clamp(max=a.T) - 1) // a.dT + 1
                d.locations_ = d.locations_[:, :, 0: a.T: a.dT]
            d.mask_ = times_ < d.lengths.view(B, A, 1) # (B, A, T)
            d.idxs_ = (batch_ * G.size + d.locations_) * T_ + times_ # (B, A, T)
            d.idxs_[~d.mask_] = 0
            d.times = times_[d.mask_]
            d.idxs = d.idxs_[d.mask_].view(-1, 1)

        dim = a.hidden_dims[0]
        obstacle = self.obstacle_embedding.weight.view(dim)

        make_grid = lambda idxs, x: obstacle.new_zeros((B * G.size * T_, self.embed_dim), requires_grad=True).scatter_add(0, idxs.expand_as(x), x)
        grid = make_grid(init.idxs, self.init_embedding(init.times))
        if shortest:
            grid = torch.cat([grid, make_grid(shortest.idxs, self.shortest_embedding(shortest.times))], dim=-1)
        grid = grid.view(B, G.size, T_, dim)
        if a.prepool > 1:
            if 'obstacles_idxs' not in a:
                a.obstacles_idxs = torch.tensor(a.obstacles, device=grid.device, dtype=torch.long)
            if grid.requires_grad: # Training
                obstacle_t = a.obstacles_idxs.view(1, -1, 1, 1).expand(B, -1, T_, dim)
                grid = grid.scatter_add(1, obstacle_t, obstacle.view(1, 1, 1, dim).expand_as(obstacle_t))
            else: # Inference
                if 'obstacles_t' not in a:
                    a.obstacles_t = grid.new_zeros((G.size, dim))
                    obstacle_idxs = a.obstacles_idxs.view(-1, 1).expand(-1, dim)
                    a.obstacles_t.scatter_add_(0, obstacle_idxs, obstacle[None, :].expand_as(obstacle_idxs))
                grid += a.obstacles_t.view(1, G.size, 1, dim)
        else:
            grid[:, a.obstacles] = obstacle
        grid[:] += self.location_embedding.weight.view(1, G.size, 1, dim)
        mask_ = init.mask_.view(B * A, T_)
        attn_mask = None
        for i, (prev_dim, dim, conv, attn) in enumerate(zip(a.hidden_dims, a.hidden_dims[1:], self.convs, self.attns)):
            grid = grid.view(B, G.rows, G.cols, T_, prev_dim)
            if conv is not None:
                grid = conv(grid.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
            grid = grid.reshape(B * G.size * T_, dim)
            x_ = grid[init.idxs_].view(B * A, T_, dim)
            if attn is not None:
                if attn_mask is None:
                    attn_mask = torch.zeros_like(mask_, dtype=grid.dtype)
                    attn_mask[~mask_] = -float('inf')
                if i < len(self.convs) - 1:
                    x = attn.feedforward_block(attn.self_attn_block(x_, attn_mask)[mask_]).to(grid.dtype)
                    if grid.requires_grad:
                        grid = grid.scatter_add(0, init.idxs.expand(-1, dim), x)
                    else:
                        grid.scatter_add_(0, init.idxs.expand(-1, dim), x)
                else:
                    x_ = attn(x_, attn_mask=attn_mask) # (B * A, T, dim)
        x_ = x_.view(B, A, T_, dim)
        x = x_[arange_B, subagents, 0] # (B, S, num_subset_agents, hidden_dims[-1])
        attn_token = self.sub_attn_embedding.weight.view(1, 1, 1, dim).expand(B, S, 1, dim)
        x = torch.cat([attn_token, x], axis=2).view(B * S, 1 + a.num_subset_agents, dim)
        x = self.sub_attns(x)[:, 0] # (B * S, hidden_dims[-1])
        x = self.out_fc(x).view(B, S)
        return self.out(x, costs)

class LinearMultiSubsetNetwork(Network):
    def __init__(self, a, train_data=None):
        super().__init__(a, train_data=train_data)
        assert a.loss != 'mse' or a.mse_clip

        num_agent_features = sum(a.num_linear_features)
        in_features = 8 * num_agent_features
        out_features = 2 if a.loss in ['bce_mse', 'svm_mse'] else 1
        self.out_fc = nn.Linear(in_features, out_features) if a.subset_fc == 'default' else nn.Sequential(*(eval('nn.' + layer) for layer in a.subset_fc))

        features_min = features_range = np.zeros(num_agent_features, dtype=np.float32)
        if train_data is not None:
            features_min, features_range = train_data['features_min'], train_data['features_range']
        self.register_buffer('features_min', torch.tensor(features_min))
        self.register_buffer('features_range', torch.tensor(features_range))

    def preprocess(self, agent_features, subagents):
        return preprocess_linear(agent_features, subagents, self.features_min, self.features_range)

    def forward(self, kwargs):
        return self.out(self.out_fc(kwargs.features).squeeze(dim=-1), kwargs.costs)

def train(train_data, val_data, net, opt, scaler, start_step, start_time, a):
    a.train_dir.mk()

    train_dataset, val_dataset = a.dataset_cls(train_data, a), a.dataset_cls(val_data, a, is_train=False)
    if a.distributed_world_size:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, drop_last=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=a.batch_train // (a.distributed_world_size or 1),
        sampler=train_sampler,
        num_workers=10,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=a.batch_eval,
        sampler=torch.utils.data.SequentialSampler(val_dataset),
        num_workers=10,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, a.train_steps, last_epoch=start_step - 1)
    logger = Logger(a.train_dir / 'train', start_time)

    start_epoch = start_step // len(train_loader)

    float16 = torch.bfloat16 if a.device == 'cpu' else torch.float16
    for epoch in itertools.count(start=start_epoch):
        if a.distributed_world_size:
            train_sampler.set_epoch(epoch)
        train_iter = iter(train_loader)
        for step in range(epoch * len(train_loader), (epoch + 1) * len(train_loader)):
            if step < start_step:
                continue
            end = step == a.train_steps

            stats = dict()
            if (step % a.save_every == 0 or end) and a.rank == 0:
                checkpoint = dict(step=step, time=time() - start_time, net=(net.module if a.distributed_world_size else net).state_dict(), opt=opt.state_dict())
                if a.use_amp: checkpoint['scaler'] = scaler.state_dict()
                torch.save(checkpoint, (a.train_dir / 'models').mk() / f'{step}.pth')

            if a.eval_every is not None and (step % a.eval_every == 0 or end) and a.rank == 0:
                print('Evaluating...')
                stats.update(evaluate(val_data, val_loader, net, a))

            if step < a.train_steps:
                get_batch_start_time = time()
                train_args = next(train_iter)

                forward_start_time = time()

                net.train()
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=a.device[:4], dtype=float16, enabled=a.use_amp):
                    out = net(in_args := to_torch(train_args, device=a.device))
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                stats.update(
                    train_loss=out.loss.item(),
                    lr=scheduler.get_last_lr()[0],
                    get_batch_time=forward_start_time - get_batch_start_time,
                    forward_backward_time=time() - forward_start_time,
                    max_memory=torch.cuda.max_memory_allocated(device=a.device),
                )
                scheduler.step()

            if a.rank == 0:
                logger.log(step, stats)
                if step % a.save_log_every == 0 or end:
                    logger.save()

            if end: return

def predict(loader, net, a):
    net.eval()
    total_loss = 0
    tensors = defaultdict(list)
    float16 = torch.bfloat16 if a.device == 'cpu' else torch.float16
    with torch.no_grad():
        for eval_args in loader:
            with torch.autocast(device_type=a.device[:4], dtype=float16, enabled=a.use_amp):
                out = net(to_torch(eval_args, device=a.device))
            total_loss += out.loss.item() * len(out.preds)
            tensors['preds'].append(out.preds.cpu())
            [tensors[k].append(v) for k, v in eval_args.costs.items()]
    return total_loss / len(loader.dataset), *(torch.cat(tensors[k]).numpy() for k in ('preds', 'final', 'shortest', 'init'))

def evaluate(data, loader, net, a):
    eval_start_time = time()
    loss, preds, final, shortest, init = predict(loader, net, a)
    true_delta = final - init
    if len(preds.shape) == 1:
        df = pd.DataFrame(dict(pred=preds, true=true_delta))
        groups = [group for iteration, group in df.groupby(data['indices'][:, 1]) if len(group) > 1]
    else:
        assert preds.shape[-1] == a.num_subsets
        groups = [pd.DataFrame(dict(pred=p, true=t)) for p, t in zip(*(x.reshape(-1, a.num_subsets) for x in [preds, true_delta]))]
    group_stats = defaultdict(list)
    for group in groups:
        group = group.sample(frac=1) # shuffle
        pred_min_idx = np.argmin(group.pred.values)
        pred_min_true = group.true.iloc[pred_min_idx]
        improvement = -min(pred_min_true, 0)
        sorted_true = np.sort(group.true)
        group_stats['improvement'].append(improvement)
        if sorted_true[0] < 0: group_stats['norm_improvement'].append(improvement / -sorted_true[0])
        group_stats['correlations'].append(group.true.corr(group.pred, method='pearson'))
        within_top_n = np.logical_or.accumulate(pred_min_true <= sorted_true)
        for i, x in enumerate(within_top_n[:10]): # up to top 10
            group_stats[f'top_{i + 1}'].append(x)
    return dict(eval_loss=loss, eval_time=time() - eval_start_time, **{k: np.nanmean(v) for k, v in group_stats.items()})

train_defaults = Namespace(
    network='Subset',
    T=34,
    dT=1,
    prepool=1,
    loss='mse',
    label_shift='init',
    label_discretization=None,
    rotation_symmetry=True,
    reflection_symmetry=False,
    mse_clip=False,
    huber_delta=None,
    features=['shortest', 'init', 'separate_static'],
    # SubsetNetwork
    kernel_size_0=5,
    padding_0=[0, 2, 0],
    location_embedding=0,
    subset_convwht='default',
    subset_convwh='default',
    subset_fc='default',
    # MultiSubsetNetwork
    hidden_dims=[8, 16, 32],
    use_3d_convolution=True,
    use_path_attention=True,
    num_heads=2,
    use_batch_norm=True,
    ff_factor=4,
    ff_factor_sub=4,
    num_subset_attention_layers=3,
    out_dims=[64, 64],
    
    reduce_data_factor=1,
    distributed_world_size=None,
    rank=0,
    device='cuda',
    checkpoint_step=None,
    train_steps=100000,
    batch_train=1024,
    batch_eval=2048,
    lr=1e-3,
    weight_decay=0, # l2 regularization
    save_every=1000,
    eval_every=2000,
    save_log_every=100,
    use_amp=False,
    use_scikit_learn=False,
    frac_sampled_problems=None,
    num_sampled_subsets=None,
    **preprocess_defaults,
)

parser = argparse.ArgumentParser()
parser.add_argument('train_dir', type=Path)
parser.add_argument('train_data_path', type=Path)
parser.add_argument('val_data_path', type=Path)

def init_train(a):
    global print
    if a.distributed_world_size:
        dist.init_process_group("nccl")
        a.rank = dist.get_rank()
        a.device = f'cuda:{a.rank % torch.cuda.device_count()}'
        print0 = print
        print = print0 if dist.get_rank() == 0 else lambda *args, **kwargs: None
    print('Training with config')
    print(format_yaml(a))

    setup_mapf(a)

    a.network_cls = eval(a.network + 'Network')
    a.dataset_cls = eval(a.network + 'Dataset')

    print('Loading in training and validation data')
    load_data_start_time = time()
    val_data = np.load(a.val_data_path)
    val_data = {k: val_data[k] for k in a.dataset_cls.data_keys or val_data.files}
    train_data = []
    if a.train_data_path is not None:
        train_data = np.load(a.train_data_path)
        train_data = [{k: train_data[k] for k in a.dataset_cls.data_keys or train_data.files}]
    print(f'Loaded data in {time() - load_data_start_time:.1f} seconds')

    net = a.network_cls(a, *train_data).to(a.device)
    if a.distributed_world_size:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DDP(net, device_ids=[a.device])

    if a.rank == 0:
        commit_dir = (a.train_dir / 'commit').mk()
        for name, info in git_state(Path(__file__).parent).items():
            (commit_dir / name + '.txt').save(info)

    opt = a.network_cls.get_opt(net, a)
    scaler = torch.cuda.amp.GradScaler(enabled=a.use_amp)
    start_step, start_time = restore(a, net, opt, scaler)
    print('Finished setup')

    a.T_data = a.T
    return *train_data, val_data, net, opt, scaler, start_step, start_time

if __name__ == '__main__':
    args = parser.parse_args()
    a = load_config(args.train_dir, parent=1).setdefaults(
        train_dir=args.train_dir,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        **train_defaults,
    )
    train(*init_train(a), a)

    a.distributed_world_size and torch.distributed.barrier()
