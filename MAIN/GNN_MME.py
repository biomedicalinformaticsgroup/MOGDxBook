import networkx as nx
import pandas as pd
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv , SAGEConv


class Encoder(torch.nn.Module):
    """
    Encoder module for the GNN_MME model.
    
    Args:
        input_dim (int): The dimensionality of the input data.
        latent_dim (int): The dimensionality of the latent space.
        output_dim (int): The dimensionality of the output data.
    """
    
    def __init__(self , input_dim , latent_dim , output_dim):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            nn.Linear(input_dim , 500), 
            nn.BatchNorm1d(500),
            nn.Linear(500, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        
        self.decoder = torch.nn.Sequential(
            nn.Linear(latent_dim, output_dim),
        )

    def forward(self, x):
        """
        Forward pass of the Encoder module.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            encoded (torch.Tensor): The encoded tensor.
            decoded (torch.Tensor): The decoded tensor.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return encoded , decoded
    
class GCN_MME(nn.Module):
    """
    Graph Convolutional Network (GCN) with Multi-Modal Encoder (MME) architecture.
    
    Args:
        input_dims (list): List of input dimensions for each modality.
        latent_dims (list): List of latent dimensions for each modality.
        decoder_dim (int): Dimension of the decoder output.
        hidden_feats (list): List of hidden feature dimensions for each layer.
        num_classes (int): Number of output classes.

    Attributes:
        encoder_dims (nn.ModuleList): List of encoder modules for each modality.
        gcnlayers (nn.ModuleList): List of GCN convolutional layers.
        batch_norms (nn.ModuleList): List of batch normalization layers.
        num_layers (int): Number of layers in the encoder.
        drop (nn.Dropout): Dropout layer.

    Methods:
        forward(g, h, subjects_list, device): Performs the forward pass of the encoder.

    """
    
    def __init__(self, input_dims, latent_dims, decoder_dim, hidden_feats, num_classes):
        super().__init__()
        
        self.encoder_dims = nn.ModuleList()
        self.gcnlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.num_layers = len(hidden_feats) + 1

        # GCN with Encoder reduced dim input and pooling scheme
        
        for modality in range(len(input_dims)):  # excluding the input layer
            self.encoder_dims.append(Encoder(input_dims[modality], latent_dims[modality], decoder_dim))
        
        for layers in range(self.num_layers):
            if layers < self.num_layers - 1:
                if layers == 0:
                    self.gcnlayers.append(
                        GraphConv(decoder_dim, hidden_feats[layers])
                    )
                else:
                    self.gcnlayers.append(
                        GraphConv(hidden_feats[layers-1], hidden_feats[layers])
                    )
            else:
                self.gcnlayers.append(
                    GraphConv(hidden_feats[layers-1], num_classes)
                )
                self.batch_norms.append(nn.BatchNorm1d(num_classes))
                
        self.drop = nn.Dropout(0.5)

    def forward(self, g, h, subjects_list, device):
        """
        Forward pass of the GCN_MME model.
        
        Args:
            g (networkx.Graph): Input graph.
            h (list): List of input node features for each modality.
            subjects_list (list): List of subject nodes for each modality.
            device (torch.device): Device to perform computations on.
        
        Returns:
            torch.Tensor: Output scores of the model.
        """
        
        reduced_dims = []
        ordered_nodes = pd.Series(nx.get_node_attributes(g, 'idx').keys()).astype(str)
        node_features = 0
        for i, Encoder in enumerate(self.encoder_dims):
            
            all_subjects = subjects_list[i] + list(set(ordered_nodes) - set(subjects_list[i]))
            reindex = pd.Series(range(len(all_subjects)), index=all_subjects).loc[ordered_nodes.values].values

            n = len(all_subjects) - len(subjects_list[i])
            encoded, decoded = Encoder(h[i])
            decoded = decoded
            decoded_imputed = torch.concat([decoded, torch.median(decoded, dim=0).values.repeat(n).reshape(n, decoded.shape[1])])[reindex]
            
            node_features += decoded_imputed
            
        node_features = node_features / (i+1)
            
        g = dgl.from_networkx(g, node_attrs=['idx', 'label']).to(device)
        g.ndata['feat'] = node_features
        
        for layers in range(self.num_layers):
            if layers == 0:
                h = self.gcnlayers[layers](g, g.ndata['feat'])
                h = self.drop(F.relu(h))
            elif layers == self.num_layers - 1:
                h = self.gcnlayers[layers](g, h)
            else:
                h = self.gcnlayers[layers](g, h)
                h = self.drop(F.relu(h))
            
        score = self.drop(h)
            
        return score
    
class GSage_MME(nn.Module):
    """
    GraphSAGE Multi-Modal Encoder (GSage_MME) class.

    This class implements a GraphSAGE-based multi-modal encoder for graph data.
    It takes in input dimensions, latent dimensions, decoder dimensions, hidden features, and number of classes as parameters.
    The forward method performs the forward pass of the encoder.

    Args:
        input_dims (list): List of input dimensions for each modality.
        latent_dims (list): List of latent dimensions for each modality.
        decoder_dim (int): Dimension of the decoder.
        hidden_feats (list): List of hidden feature dimensions for each layer.
        num_classes (int): Number of output classes.

    Attributes:
        encoder_dims (nn.ModuleList): List of encoder modules for each modality.
        gnnlayers (nn.ModuleList): List of GraphSAGE convolutional layers.
        batch_norms (nn.ModuleList): List of batch normalization layers.
        num_layers (int): Number of layers in the encoder.
        drop (nn.Dropout): Dropout layer.

    Methods:
        forward(g, h, subjects_list, device): Performs the forward pass of the encoder.

    """

    def __init__(self, input_dims, latent_dims, decoder_dim, hidden_feats, num_classes):
        super().__init__()

        self.encoder_dims = nn.ModuleList()
        self.gnnlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.num_layers = len(hidden_feats) + 1

        # GCN with Encoder reduced dim input and pooling scheme
        for modality in range(len(input_dims)):  # excluding the input layer
            self.encoder_dims.append(Encoder(input_dims[modality], latent_dims[modality], decoder_dim))

        for layers in range(self.num_layers):
            if layers < self.num_layers - 1:
                if layers == 0:
                    self.gnnlayers.append(
                        SAGEConv(decoder_dim, hidden_feats[layers], 'pool', feat_drop=0.1)
                    )
                else:
                    self.gnnlayers.append(
                        SAGEConv(hidden_feats[layers - 1], hidden_feats[layers], 'pool', feat_drop=0.1)
                    )
                self.batch_norms.append(nn.BatchNorm1d(hidden_feats[layers]))
            else:
                self.gnnlayers.append(
                    SAGEConv(hidden_feats[layers - 1], num_classes, 'pool', feat_drop=0.1)
                )

        self.drop = nn.Dropout(0.5)

    def forward(self, g, h, subjects_list, device):
        """
        Performs the forward pass of the GSage_MME encoder.

        Args:
            g (networkx.Graph): Input graph.
            h (list): List of input node features for each modality.
            subjects_list (list): List of subjects for each modality.
            device (torch.device): Device to perform computation on.

        Returns:
            h (torch.Tensor): Output node features after the forward pass.

        """
        reduced_dims = []
        ordered_nodes = pd.Series(nx.get_node_attributes(g, 'idx')).astype(str)
        node_features = 0
        for i, Encoder in enumerate(self.encoder_dims):

            all_subjects = subjects_list[i] + list(set(ordered_nodes) - set(subjects_list[i]))
            reindex = pd.Series(range(len(all_subjects)), index=all_subjects).loc[ordered_nodes.values].values

            n = len(all_subjects) - len(subjects_list[i])
            encoded, decoded = Encoder(h[i])
            decoded = self.drop(decoded)
            decoded_imputed = torch.concat([decoded, torch.median(decoded, dim=0).values.repeat(n).reshape(n, decoded.shape[1])])[reindex]

            node_features += decoded_imputed

        node_features = node_features / (i + 1)

        g = dgl.from_networkx(g).to(device)
        g.ndata['feat'] = node_features
        if g.in_degrees().sum() == 0:
            g = dgl.add_self_loop(g)

        for layers in range(self.num_layers):
            if layers == 0:
                h = self.gnnlayers[layers](g, g.ndata['feat'])
                h = F.relu(h)
            elif layers == self.num_layers - 1:
                h = self.gnnlayers[layers](g, h)
            else:
                h = self.gnnlayers[layers](g, h)
                h = F.relu(h)

        return h