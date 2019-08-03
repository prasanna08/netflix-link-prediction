import torch
import torch.nn.functional as F
device = ('cuda' if torch.cuda.is_available() else 'cpu')

class GraphLayer(torch.nn.Module):
    def __init__(self, features, dropout):
        super(GraphLayer, self).__init__()
        self.features = features
        self.usr_conv = torch.nn.Conv1d(self.features, self.features, kernel_size=1)
        self.mvi_conv = torch.nn.Conv1d(self.features, self.features, kernel_size=1)
        self.usr_drop = torch.nn.Dropout(dropout)
        self.mvi_drop = torch.nn.Dropout(dropout)
        self.usr_layer_norm = torch.nn.LayerNorm(features)
        self.mvi_layer_norm = torch.nn.LayerNorm(features)
        self.up_usr = torch.nn.Linear(2*self.features, self.features)
        self.up_mvi = torch.nn.Linear(2*self.features, self.features)
    
    def forward(self, adj, usr, mvi):
        nusr = self.usr_drop(usr)
        nmvi = self.mvi_drop(mvi)
        nusr = self.usr_conv(nusr.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        nmvi = self.mvi_conv(nmvi.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        nnmvi = torch.mm(adj.transpose(0, 1), nusr)
        nnusr = torch.mm(adj, nmvi)
        nnmvi = nnmvi * torch.sigmoid(nnmvi)
        nnusr = nnusr * torch.sigmoid(nnusr)
        upusr = torch.sigmoid(self.up_usr(torch.cat([nnusr, usr], dim=1)))
        upmvi = torch.sigmoid(self.up_mvi(torch.cat([nnmvi, mvi], dim=1)))
        nnmvi = self.mvi_layer_norm(upmvi * nnmvi + (1 - upmvi) * mvi)
        nnusr = self.usr_layer_norm(upusr * nnusr + (1 - upusr) * usr)
        return nnusr, nnmvi

class GraphVAE(torch.nn.Module):
    def __init__(self, nhidden, adj, user_to_idx, movie_to_idx, dropout, num_layers, vae_mode):
        super(GraphVAE, self).__init__()
        self.nhidden = nhidden
        self.adj = adj
        self.nusers = len(user_to_idx)
        self.nmovies = len(movie_to_idx)
        self.graph_layers = [GraphLayer(nhidden, dropout) for _ in range(num_layers)]
        self.mean_conv = torch.nn.Conv1d(nhidden, nhidden, kernel_size=1)
        self.log_var_conv = torch.nn.Conv1d(nhidden, nhidden, kernel_size=1)
        self.combined_layer = torch.nn.Linear(2*nhidden, nhidden)
        self.output_layer = torch.nn.Linear(nhidden, 5)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.user_embed = torch.nn.Parameter(torch.empty(self.nusers, nhidden))
        self.movie_embed = torch.nn.Parameter(torch.empty(self.nmovies, nhidden))
        self.user_embed = torch.nn.Linear(self.nmovies, nhidden)
        self.movie_embed = torch.nn.Linear(self.nusers, nhidden)
        self.mvi_init = torch.nn.Conv1d(self.nusers, nhidden, kernel_size=1)
        for i, graph_layer in enumerate(self.graph_layers):
            self.add_module(module=graph_layer, name='Graph-%d' % i)
        torch.nn.init.xavier_normal_(self.user_embed)
        torch.nn.init.xavier_normal_(self.movie_embed)
        self.vae_mode = vae_mode

    def forward(self, usr_idx, mvi_idx):
        usr = self.user_embed(self.adj)
        mvi = self.movie_embed(self.adj.transpose(0, 1))
        for layer in self.graph_layers:
            usr, mvi = layer(self.adj, usr, mvi)
        zusr = usr
        zmvi = mvi
        if self.vae_mode:
            musr = self.mean_conv(zusr).squeeze(0).transpose(0, 1)
            mmvi = self.mean_conv(zmvi).squeeze(0).transpose(0, 1)
            lvusr = self.log_var_conv(zusr).squeeze(0).transpose(0, 1)
            lvmvi = self.log_var_conv(zmvi).squeeze(0).transpose(0, 1)
            musr = musr * torch.sigmoid(musr)
            mmvi = mmvi * torch.sigmoid(mmvi)
            lvusr = lvusr * torch.sigmoid(lvusr)
            lvmvi = lvmvi * torch.sigmoid(lvmvi)
            epsusr = torch.randn_like(lvusr, requires_grad=False)
            epsmvi = torch.randn_like(lvmvi, requires_grad=False)
            dusr = musr + torch.exp(0.5*lvusr) * epsusr
            dmvi = mmvi + torch.exp(0.5*lvmvi) * epsmvi
            usrs = dusr[usr_idx]
            mvis = dmvi[mvi_idx]
        else
            usrs = zusr[usr_idx]
            mvis = zmvi[mvi_idx]
        pvec = self.combined_layer(torch.cat([usrs, mvis], dim=1))
        outs = self.output_layer(pvec)
        klloss = None
        if self.vae_mode:
            klloss = -0.5 * (torch.sum(1 + lvusr - musr.pow(2) - lvusr.exp()) + torch.sum(1 + lvmvi - mmvi.pow(2) - lvmvi.exp()))
        return self.log_softmax(outs), klloss
