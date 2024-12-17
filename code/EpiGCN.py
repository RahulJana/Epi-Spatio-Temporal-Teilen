# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.utils import dense_to_sparse

class SSIR_STGCN(nn.Module):
    """
    SSIR-STGCN Model integrating SIR epidemic modeling with Spatio-Temporal Graph Convolutions.
    
    Args:
        obs_len (int): Length of the observation window.
        pre_len (int): Length of the prediction horizon.
        kernel_size (int): Kernel size for temporal convolutions.
        num_layers (int): Number of layers in graph convolutions.
        adj_type (str): Type of adjacency matrix ('Static' or 'Dynamic').
        neighbor_matrix (torch.Tensor): Predefined adjacency matrix for static graphs.
        in_dim (int, optional): Dimension of input features. Defaults to 3.
        nsize (int, optional): Number of nodes in the graph. Defaults to 31.
        t_out_dim (int, optional): Output dimension of temporal embeddings. Defaults to 16.
        s_out_dim (int, optional): Output dimension of spatial embeddings. Defaults to 16.
        dropout (float, optional): Dropout rate. Defaults to 0.5.
        beta_incorporated (bool, optional): Whether to incorporate beta in SIR equations. Defaults to False.
    """
    def __init__(
        self,
        obs_len,
        pre_len,
        kernel_size,
        num_layers,
        adj_type,
        neighbor_matrix,
        in_dim,
        t_out_dim,
        s_out_dim,
        dropout,
        nsize,
        beta_incorporated=False,
    ):
        super(SSIR_STGCN, self).__init__()
        
        self.adj_type = adj_type
        self.fc = nn.Sequential(
            nn.Linear(in_dim, t_out_dim),
            nn.ReLU()
        )
        self.graph_module = GraphModule(adj_type, nsize, t_out_dim, neighbor_matrix)
        self.temporal_module = TemporalModule(kernel_size, obs_len, obs_len, t_out_dim, nsize)
        self.spatial_module = SpatialModule(t_out_dim, s_out_dim, num_layers, dropout, adj_type)
        self.epidemic_module = EpidemicModule(obs_len, s_out_dim, pre_len, nsize, beta_incorporated)
        self.forecasting_module = ForecastingModule(s_out_dim, pre_len)
    
    def forward(self, x):
        """
        Forward pass of the SSIR-STGCN model.
        
        Args:
            x (Tensor): Input tensor of shape (B, T_obs, N, F).
        
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
                - y_pre: Forecasted values (B, T_next, N, 1)
                - y_phy: Physical model predictions (B, T_next, N, 1)
                - epiparams: Epidemic parameters (B, T_next, N, N + 2)
                - all_phy: Full SIR states (B, T_next, N, 3)
                - adjmatrix: Adjacency matrix (only if adj_type is 'Dynamic')
        """
        # H=FC(X), mapping the epidemic data to a high dimensional time-series embedding.
        # (B,T_obs,N,F(3)) -> (B,T_obs,N,F(embedding))
        x_fc = self.fc(x)

        # Temporal feature extraction
        # (B,T_obs,N,F(embedding)) -> (B,T_obs,N,F(t_embedding))
        x_temporal = self.temporal_module(x_fc)
        
        # Adjacency matrix computation
        # If Static: (N, N), if Dynamic: (B, T, N, N)
        adjmatrix = self.graph_module(x_fc)

        # Spatial feature extraction
        # (B,T_obs,N,F(t_embedding)) -> (B,T_obs,N,F(s_embedded))
        x_spatial = self.spatial_module(x_temporal, adjmatrix)

        # Forecasting
        # (B,T_obs,N,F(s_embedded)) -> (B,T_next,N,1)
        y_pre = self.forecasting_module(x_spatial)

        # Epidemic model predictions
        # (B,T_obs,N,F(s_embedded)), (B,T_obs,N,F) -> (B,T_next,N,1), (B,T_next,N, N + 2), (B,T_next,N, 3)
        y_phy, epiparams, all_phy, y_phy_d = self.epidemic_module(x_spatial, x)

        # Conditional return based on adjacency type
        return y_pre, y_phy, epiparams, all_phy, y_phy_d, (adjmatrix if self.adj_type == 'Dynamic' else None)

class TemporalModule(nn.Module):
    """
    Temporal feature extraction module using series decomposition.
    
    Args:
        kernel_size (int): Kernel size for moving average.
        obs_len (int): Length of the observation window.
        pre_len (int): Length of the prediction horizon.
        channels (int): Number of feature channels.
        nsize (int): Number of nodes.
        separate (bool, optional): Whether to apply separate linear layers for each channel. Defaults to None.
    """
    def __init__(self, kernel_size, obs_len, pre_len, channels, nsize, separate=False):
        super(TemporalModule, self).__init__()
        self.in_len = obs_len
        self.out_len = pre_len  # Input and output sizes are the same
        
        self.series_decomposition = SeriesDecomposition(kernel_size)
        self.separate = separate
        self.channels = channels  # Number of features
        
        if self.separate:
            self.linear_trend = nn.ModuleList()
            self.norm_trend = nn.ModuleList()
            self.linear_remain = nn.ModuleList()
            self.norm_remain = nn.ModuleList()

            for _ in range(self.channels):
                self.linear_trend.append(nn.Linear(self.in_len, self.out_len))
                self.norm_trend.append(nn.LayerNorm([nsize, self.out_len]))
                self.linear_remain.append(nn.Linear(self.in_len, self.out_len))
                self.norm_remain.append(nn.LayerNorm([nsize, self.out_len]))
        else:
            self.linear_trend = nn.Linear(self.in_len, self.out_len)
            self.norm_trend = nn.LayerNorm([nsize, self.out_len])
            self.linear_remain = nn.Linear(self.in_len, self.out_len)
            self.norm_remain = nn.LayerNorm([nsize, self.out_len])
        
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the TemporalModule.
        
        Args:
            x (Tensor): Input tensor of shape (B, T, N, F).
        
        Returns:
            Tensor: Output tensor of shape (B, T, N, F).
        """
        # (B,T,N,F) -> (B,T,N,F)
        res = x
        
        batch, _, node, feature = x.size()
        
        # Decompose into trend and remain
        trend_init, remain_init = self.series_decomposition(x)
        # (B,T,N,F) -> (B,F,N,T)
        trend_init = trend_init.permute(0, 3, 2, 1)
        remain_init = remain_init.permute(0, 3, 2, 1)

        if self.separate:
            # Initialize outputs
            trend_output = torch.zeros(batch, feature, node, self.out_len, dtype=x.dtype, device=x.device)
            remain_output = torch.zeros(batch, feature, node, self.out_len, dtype=x.dtype, device=x.device)

            for i in range(self.channels):
                # Apply linear transformation and normalization per channel
                trend = self.linear_trend[i](trend_init[:, i, :, :])  # (B, N, out_len)
                trend = self.norm_trend[i](trend)  # (B, N, out_len)
                trend_output[:, i, :, :] = self.relu(trend)

                remain = self.linear_remain[i](remain_init[:, i, :, :])  # (B, N, out_len)
                remain = self.norm_remain[i](remain)  # (B, N, out_len)
                remain_output[:, i, :, :] = self.relu(remain)
        else:
            trend_output = self.relu(self.norm_trend(self.linear_trend(trend_init)))
            remain_output = self.relu(self.norm_remain(self.linear_remain(remain_init)))
        
        # Combine trend and remain
        x_cat = trend_output + remain_output  # (B, F, N, T)
        
        # Permute back to (B, T, N, F)
        x_cat = x_cat.permute(0, 3, 2, 1)
        
        # Residual connection
        if res.size() == x_cat.size():
            x_cat = x_cat + res

        return x_cat

class SeriesDecomposition(nn.Module):
    """
    Series decomposition into trend and remainder using moving average.
    
    Args:
        kernel_size (int): Kernel size for moving average.
    """
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_average = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        """
        Forward pass of the SeriesDecomposition.
        
        Args:
            x (Tensor): Input tensor of shape (B, T, N, F).
        
        Returns:
            Tuple[Tensor, Tensor]: Trend and remainder tensors, each of shape (B, T, N, F).
        """
        # (B,T,N,F) -> (B,T,N,F)
        trend = self.moving_average(x)
        
        # (B,T,N,F) -> (B,T,N,F)
        remain = x - trend 

        return trend, remain

class MovingAverage(nn.Module):
    """
    Moving average layer using 1D average pooling.
    
    Args:
        kernel_size (int): Size of the moving window.
        stride (int): Stride for the pooling operation.
    """
    def __init__(self, kernel_size, stride):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        # Ensure symmetric padding
        self.avg = nn.AvgPool1d(
            kernel_size=self.kernel_size,
            stride=stride,
            padding=(self.kernel_size - 1) // 2,
        )

    def forward(self, x):
        """
        Forward pass of the MovingAverage.
        
        Args:
            x (Tensor): Input tensor of shape (B, T, N, F).
        
        Returns:
            Tensor: Moving averaged tensor of shape (B, T, N, F).
        """
        batch, time, node, feature = x.size()
        
        # Permute to (B, N*F, T) for AvgPool1d
        x_view = x.permute(0, 2, 3, 1).contiguous().view(batch, node * feature, time)
        
        # Apply average pooling
        x_avg = self.avg(x_view)  # (B, N*F, T)
        
        # Reshape back to (B, T, N, F)
        x_avg_deview = x_avg.view(batch, node, feature, time).permute(0, 3, 1, 2)
        
        return x_avg_deview

class SpatialModule(nn.Module):
    """
    Spatial feature extraction module using multi-layer graph convolutions.
    
    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        num_layers (int): Number of graph convolution layers.
        dropout (float): Dropout rate.
        adjmatrix_type (str): Type of adjacency matrix ('Static' or 'Dynamic').
    """
    def __init__(self, in_dim, out_dim, num_layers, dropout, adjmatrix_type):
        super(SpatialModule, self).__init__()
        self.out_dim = out_dim
        self.adjmatrix_type = adjmatrix_type
        
        self.mgcn = MultiGraphConvolution(in_dim, out_dim, num_layers, dropout)
        
        self.edge_index = None
        self.edge_weight = None

    def forward(self, x, adjacency):
        """
        Forward pass of the SpatialModule.
        
        Args:
            x (Tensor): Input feature tensor of shape (B, T, N, F).
            adjacency (Tensor): Adjacency matrix tensor.
                - If 'Static': shape (N, N).
                - If 'Dynamic': shape (B, T, N, N).
        
        Returns:
            Tensor: Output feature tensor of shape (B, T, N, out_dim).
        """
        batch_size, time_steps, num_nodes, feature_dim = x.size()
        x_view = x.contiguous().view(batch_size * time_steps, num_nodes, feature_dim)

        if self.adjmatrix_type == 'Static':
            if self.edge_index is None or self.edge_weight is None:
                edge_index, edge_weight = dense_to_sparse(adjacency)
                self.edge_index = edge_index
                self.edge_weight = edge_weight

            # Create Data objects for static adjacency
            data_list = [Data(x=x_view[i], edge_index=self.edge_index, edge_attr=self.edge_weight) for i in range(batch_size * time_steps)]
            batch_graph = Batch.from_data_list(data_list)

            # Apply GCN
            gcn_output = self.mgcn(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr, batch_graph.batch)
            
            # Reshape back to (B, T, N, out_dim)
            gcn_output = gcn_output.view(batch_size, time_steps, num_nodes, self.out_dim)
        else:
            # Dynamic adjacency matrix: (B, T, N, N)
            adjacency = adjacency.view(batch_size * time_steps, num_nodes, num_nodes)
            # data_list = []
            edge_indices = []
            edge_weights = []
            for i in range(batch_size * time_steps):
                adj_i = adjacency[i]
                edge_index_i, edge_weight_i = dense_to_sparse(adj_i)
                edge_indices.append(edge_index_i)
                edge_weights.append(edge_weight_i)
            
            # Batch all graphs
            edge_index = torch.cat(edge_indices, dim=1)
            edge_weight = torch.cat(edge_weights)
            
            batch = torch.arange(batch_size * time_steps).repeat_interleave(num_nodes).to(x.device)

            # Apply GCN
            x_flat = x_view.view(-1, feature_dim)
            gcn_output = self.mgcn(x_flat, edge_index, edge_weight, batch)
            
            # Reshape back to (B, T, N, out_dim)
            gcn_output = gcn_output.view(batch_size, time_steps, num_nodes, self.out_dim)

        return gcn_output

class MultiGraphConvolution(nn.Module):
    """
    Multi-layer Graph Convolutional Network with residual connections and dropout.
    
    Args:
        input_dim (int): Input feature dimension.
        output_dim (int): Output feature dimension.
        num_layers (int): Number of GCN layers.
        dropout (float): Dropout rate.
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 64.
    """
    def __init__(self, input_dim, output_dim, num_layers, dropout, hidden_dim=64):
        super(MultiGraphConvolution, self).__init__()
        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_channels = input_dim
        self.residual_transforms = nn.ModuleList()

        for i in range(num_layers):
            self.gconvs.append(GCNConv(in_channels, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))
            if in_channels != hidden_dim:
                self.residual_transforms.append(nn.Linear(in_channels, hidden_dim))
            else:
                self.residual_transforms.append(nn.Identity())
            in_channels = hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        if hidden_dim != output_dim:
            self.final_residual_transform = nn.Linear(input_dim, output_dim)
        else:
            self.final_residual_transform = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight, batch):
        """
        Forward pass of the MultiGraphConvolution.
        
        Args:
            x (Tensor): Input features of shape (Total Nodes, input_dim).
            edge_index (Tensor): Edge indices for all graphs.
            edge_weight (Tensor): Edge weights.
            batch (Tensor, optional): Batch vector assigning each node to a specific graph.
        
        Returns:
            Tensor: Output features of shape (Total Nodes, output_dim).
        """
        res_init = x
        for i in range(self.num_layers):
            res = x
            x = self.gconvs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = self.relu(x)
            if i < self.num_layers - 1:
                x = self.dropout(x)
            
            res_transformed = self.residual_transforms[i](res)
            x = x + res_transformed
        x = self.fc(x)
        x = x + self.final_residual_transform(res_init)
        
        return x

class GraphModule(nn.Module):
    """
    Module to compute adjacency matrices, either static or dynamic.
    
    Args:
        adj_type (str): Type of adjacency matrix ('Static' or 'Dynamic').
        nsize (int): Number of nodes.
        out_dim (int): Output dimension for embeddings.
        neighbor_matrix (torch.Tensor): Predefined adjacency matrix for static graphs.
    """
    def __init__(self, adj_type, nsize, out_dim, neighbor_matrix):
        super(GraphModule, self).__init__()
        self.adj_type = adj_type
        self.neighbor_matrix = neighbor_matrix
        self.relu = nn.ReLU()
        
        if self.adj_type == 'Dynamic': 
            # Temporal Convolutional Network for dynamic adjacency generation
            self.tcn = TCNModule(in_channels=out_dim, out_channels=out_dim)
            
            self.embedding = nn.Parameter(torch.randn(nsize, out_dim), requires_grad=True)
            nn.init.xavier_uniform_(self.embedding)
            
        if self.adj_type == 'Adaptive':
            # Placeholder for adaptive adjacency matrix
            self.adaptive_matrix = nn.Parameter(torch.randn(1, 1, nsize, nsize), requires_grad=True)
            nn.init.xavier_uniform_(self.adaptive_matrix)

    def forward(self, x):
        """
        Forward pass of the GraphModule.
        
        Args:
            x (Tensor): Input tensor of shape (B, T, N, F).
        
        Returns:
            Tensor: Adjacency matrix tensor.
                - If 'Static': shape (N, N).
                - If 'Dynamic': shape (B, T, N, N).
                - If 'Adaptive': shape (B, T, N, N).
        """
        if self.adj_type == 'Static':            
            return self.neighbor_matrix
        
        elif self.adj_type == 'Adaptive':
            adaptive_adj = self.adaptive_matrix.expand(x.size(0), x.size(1), x.size(2), x.size(2))
            return self.relu(adaptive_adj)
        
        else:
            # Compute static adjacency based on embeddings
            adj_back = torch.matmul(self.embedding, self.embedding.T).to(x.device)  # (N, N)
            adj_back = self.relu(adj_back)
            # adj_back = F.softmax(adj_back, dim=-1)  # Normalize along node dimension
            
            # Compute dynamic adjacency via TCN
            adj_tcn = self.tcn(x)  # (B, T, N, F)
            
            # Generate dynamic adjacency matrix
            adj_tcn_transpose = adj_tcn.permute(0, 1, 3, 2)  # (B, T, F, N)
            adj_temp = torch.matmul(adj_tcn, adj_tcn_transpose)  # (B, T, N, N)
            
            adj_temp = F.relu(adj_temp)
            # adj_temp = F.softmax(adj_temp, dim=-1)  # Normalize along node dimension
            
            # Combine static and dynamic adjacency matrices
            adj_combined = adj_back.unsqueeze(0).unsqueeze(0) + adj_temp  # Broadcast to (B, T, N, N)
            adj = F.softmax(adj_combined, dim=-1)  # Final normalization
            
            return adj

class TCNModule(nn.Module):
    """
    Temporal Convolutional Network for dynamic adjacency matrix generation.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size for convolutions. Defaults to 3.
        dilation (int, optional): Dilation factor. Defaults to 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNModule, self).__init__()
        self.tcn = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            dilation=dilation, 
            padding=(kernel_size - 1) * dilation // 2
        )

    def forward(self, x):
        """
        Forward pass of the TCNModule.
        
        Args:
            x (Tensor): Input tensor of shape (B, T, N, F).
        
        Returns:
            Tensor: Output tensor of shape (B, N, F, T).
        """
        batch, time, node, feature = x.size()
        
        # Permute to (B, N, F, T) for Conv1d
        x_view = x.permute(0, 2, 3, 1).contiguous().view(batch * node, feature, time)
        
        # Apply TCN layer
        x_t = self.tcn(x_view)  # (B*N, out_channels, T)
        
        # Reshape back to (B, T, N, out_channels)
        x_deview = x_t.view(batch, node, -1, time).permute(0, 3, 1, 2)
        
        return x_deview  # (B, T, N, F)

class EpidemicModule(nn.Module):
    """
    Neural network-based SIR epidemic model module.
    
    Args:
        obs_len (int): Length of the observation window.
        in_dim (int): Dimension of input features.
        pre_len (int): Length of the prediction horizon.
        nsize (int): Number of nodes (used for potential scaling).
        beta_incorporated (bool): Flag to determine beta incorporation in SIR equations.
    """
    def __init__(self, obs_len, in_dim, pre_len, nsize, beta_incorporated, hidden_dim=64):
        super(EpidemicModule, self).__init__()
        self.obs_len = obs_len
        self.pre_len = pre_len
        self.nsize = nsize
        self.beta_incorporated = beta_incorporated

        self.beta_lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.beta_fc = nn.Sequential(
            nn.Linear(hidden_dim, pre_len),
            nn.Sigmoid()
        )

        self.gamma_lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.gamma_fc = nn.Sequential(
            nn.Linear(hidden_dim, pre_len),
            nn.Sigmoid()
        )

        self.cij_lstm = nn.LSTM(input_size=2 * in_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.cij_fc = nn.Sequential(
            nn.Linear(hidden_dim, pre_len),
            nn.Sigmoid()
        )

    def forward(self, x_s, x):
        """
        Forward pass of the EpidemicModule.
        
        Args:
            x_s (Tensor): Spatial feature tensor of shape (B, T_obs, N, F).
            x (Tensor): Original input tensor of shape (B, T_obs, N, F).
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - y_phy: Physical model predictions (B, T_next, N, 1)
                - epiparams: Epidemic parameters (B, T_next, N, N + 2)
                - all_phy: Full SIR states (B, T_next, N, 3)
        """
        batch, time, node, feature = x_s.size()

        # (B*N, T_obs, F)
        x_s_view = x_s.permute(0, 2, 1, 3).contiguous().view(batch * node, time, feature)

        # LSTM  beta
        beta_lstm_out, _ = self.beta_lstm(x_s_view)  # (B*N, T_obs, hidden_dim)
        beta_hidden = beta_lstm_out[:, -1, :]  # take the last time step of hiddern feature
        betas = self.beta_fc(beta_hidden).view(batch, node, self.pre_len, 1)  # (B, N, pre_len, 1)

        # LSTM  gamma
        gamma_lstm_out, _ = self.gamma_lstm(x_s_view)
        gamma_hidden = gamma_lstm_out[:, -1, :]
        gammas = self.gamma_fc(gamma_hidden).view(batch, node, self.pre_len, 1)  # (B, N, pre_len, 1)

        # x_s: (B, T_obs, N, F)
        x_s_transpose = x_s.permute(0, 2, 1, 3)  # (B, N, T_obs, F)

        H_i = x_s_transpose.unsqueeze(2).expand(-1, -1, node, -1, -1)  # (B, N, N, T_obs, F)
        H_j = x_s_transpose.unsqueeze(1).expand(-1, node, -1, -1, -1)  # (B, N, N, T_obs, F)

        edge_features = torch.cat((H_i, H_j), dim=-1)  # (B, N, N, T_obs, 2F)
        edge_features = edge_features.view(batch * node * node, time, 2 * feature)  # (B*N*N, T_obs, 2F)

        #  LSTM  c_ij
        cij_lstm_out, _ = self.cij_lstm(edge_features)  # (B*N*N, T_obs, hidden_dim)
        cij_hidden = cij_lstm_out[:, -1, :]  # take the last time step of hiddern feature
        c_ij = self.cij_fc(cij_hidden).view(batch, node, node, self.pre_len)  # (B, N, N, pre_len)
        c_ij = c_ij.permute(0, 3, 1, 2)  # (B, pre_len, N, N)

        c_ij = F.softmax(c_ij, dim=-1)  # (B, pre_len, N, N)
        cs = c_ij.permute(0, 2, 1, 3)  # (B, N, pre_len, N)

        x_last = x[:, -1, :, :]  # (B, N, F)
        output = torch.zeros(
            batch,
            node,
            self.pre_len,
            x_last.size(-1)+1, # I_new,S,I,R
            dtype=x_last.dtype,
            device=x_last.device,
        )
        
        for t in range(self.pre_len):
            nsir = self.sircell(
                betas[:, :, t, :],
                gammas[:, :, t, :],
                cs[:, :, t, :],
                x_last
            )
            x_last = nsir[:, :, 1:]  # (B, N, [S, I, R])
            output[:, :, t, :] =  nsir[:, :, :]   # (B, N, pre_len, [I_new, S, I, R])
        
        epiparams = torch.cat([betas, gammas, cs], dim=-1)  # (B, N, pre_len, N + 2)
        epiparams_view = epiparams.permute(0, 2, 1, 3)   # (B, pre_len, N, N + 2)
        output_view = output.permute(0, 2, 1, 3)        # (B, pre_len, N, 4)

        # I, epi_params, SIR, Daily
        return output_view[:, :, :, 2:3], epiparams_view, output_view[:, :, :, 1:], output_view[:, :, :, 0:1]
    
    def sircell(self, beta, gamma, c, sir):
        """
        SIR cell computation.
        
        Args:
            beta (Tensor): Infection rate tensor of shape (B, N, 1).
            gamma (Tensor): Recovery rate tensor of shape (B, N, 1).
            c (Tensor): Connection coefficients tensor of shape (B, N, N).
            sir (Tensor): Current SIR state tensor of shape (B, N, 3).
        
        Returns:
            Tensor: Updated SIR state tensor of shape (B, N, 4).
        """
        S, I, R = sir[:, :, 0:1], sir[:, :, 1:2], sir[:, :, 2:3]  # (B, N, 1)
        
        epsilon = 1e-8  # Prevent division by zero
        N = (S + I + R).clamp(min=epsilon)  # (B, N, 1)
        
        infection_term = torch.matmul(c, I)  # (B, N, 1)
        
        if self.beta_incorporated:
            delta_S = -beta / N * infection_term
        else:
            delta_S = -beta * S / N * infection_term
        
        delta_I = -delta_S - gamma * I
        delta_R = gamma * I
        
        # Ensure non-negativity
        S_t = torch.clamp(S + delta_S, min=0.0)
        I_t = torch.clamp(I + delta_I, min=0.0)
        R_t = torch.clamp(R + delta_R, min=0.0)
        
        # normalizaion
        total = S_t + I_t + R_t
        scaling_factor = N / total.clamp(min=epsilon)
        S_t = S_t * scaling_factor
        I_t = I_t * scaling_factor
        R_t = R_t * scaling_factor
        
        I_new = torch.clamp(-delta_S, min=0.0)  # Daily new infections
        
        return torch.cat((I_new, S_t, I_t, R_t), dim=2)  # (B, N, 4)
    

class ForecastingModule(nn.Module):
    def __init__(self, in_dim, pre_len, hidden_dim=32, num_layers=3):
        super(ForecastingModule, self).__init__()
        
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.pre_len = pre_len

    def forward(self, x):
        batch, time, node, feature = x.size()
        
        x = x.permute(0, 2, 1, 3)  # (B, N, T, F)
        x = x.reshape(batch * node, time, feature)
        
        lstm_out, _ = self.lstm(x)
        # last time step
        lstm_out = lstm_out[:, -1, :]
        x_fc = self.fc(lstm_out)

        x_fc = F.relu(x_fc)
        x_fc = x_fc.view(batch, node, 1)
        
        # forecasting for pre_len times
        output = x_fc.unsqueeze(1).repeat(1, self.pre_len, 1, 1)  # (B, pre_len, N, 1)
        
        return output
