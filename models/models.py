import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.init import xavier_normal_
from .odeblock import ODEBlock
from .anieblock import ANIE
from .MGCN import *
from .MGCNLayer import *
from .RGAT import *
from IE_source.Galerkin_transformer import RGAT, RelationalTransformer
from collections import defaultdict

class TANGO(nn.Module):
    def __init__(self, num_e, num_rel, params, device, logger):
        super().__init__()

        self.num_e = num_e
        self.num_rel = num_rel
        self.p = params
        self.core = self.p.core
        self.core_layer = self.p.core_layer
        self.score_func = self.p.score_func
        self.device = self.p.device
        self.initsize = self.p.initsize
        self.drop = self.p.dropout
        self.hidsize = self.p.hidsize
        self.embsize = self.p.embsize

        self.device = device
        self.logger = logger
        if self.p.activation.lower() == 'tanh':
            self.act = torch.tanh
        elif self.p.activation.lower() == 'relu':
            self.act = F.relu
        elif self.p.activation.lower() == 'leakyrelu':
            self.act = F.leaky_relu

        # define loss
        self.loss = torch.nn.CrossEntropyLoss()

        # define entity and relation embeddings
        self.emb_e = self.get_param((self.num_e, self.initsize))
        self.emb_r = self.get_param((self.num_rel * 2, self.initsize))

        # define graph ode core
        # self.gde_func = self.construct_gde_func()

        # define ode block
        # self.odeblock = self.construct_GDEBlock(self.gde_func)

        # define jump modules
        if self.p.jump:
            self.jump, self.jump_weight = self.Jump()
            self.gde_func.jump = self.jump
            self.gde_func.jump_weight = self.jump_weight

        # score function TuckER
        if self.score_func.lower() == "tucker":
            self.W_tk, self.input_dropout, self.hidden_dropout1, self.hidden_dropout2, self.bn0, self.bn1 = self.TuckER()
        else:
            self.W, self.b = self.Boy()
        self.graph_feature = self.add_base()
        
        config = defaultdict(lambda: None,
                    #embedding
                    node_feats=self.p.node_feats,
                    edge_feats=self.p.edge_feats,
                    pos_dim=self.p.pos_dim,    
                    #RGAT
                    feat_extract_type='rgat',
                    n_hidden=self.p.n_hidden, 
                    graph_activation=True,
                    num_feat_layers=2,
                    raw_laplacian=False,               
                    sp_type=False,
                    # SimpleTransformer 
                    num_encoder_layers=self.p.num_encoder_layers,
                    n_head=self.p.n_head,
                    dim_feedforward=self.p.dim_feedforward,
                    attention_type=self.p.attention_type,  # no softmax ['fourier', 'integral', 'cosine', 'galerkin', 'linear', 'softmax']
                    xavier_init=self.p.xavier_init,
                    diagonal_weight=self.p.diagonal_weight,    #for attention
                    symmetric_init=self.p.symmetric_init,
                    layer_norm=True,
                    attn_norm=False,
                    batch_norm=False,
                    spacial_residual=False,
                    return_attn_weight=True,
                    # freq BulkRegressor
                    pred_len=0,             #
                    n_freq_targets=0,       #
                    seq_len=None,           # seq_len = input_step + delat_step
                    bulk_regression=False,
                    # SpectralRegressor
                    decoder_type=self.p.decoder_type,
                    n_targets=self.p.n_targets,           #out_dim
                    freq_dim=self.p.freq_dim,
                    num_regressor_layers=self.p.num_regressor_layers,
                    fourier_modes=self.p.fourier_modes,           # modes <= ceil(input_step+delta_step+1)
                    debug=self.p.dropout,
                    spacial_dim=1,
                    spacial_fc=False,
                    dropout=0.1,            
                    )
        self.ft = RelationalTransformer(**config)
        self.ANIEBlock = ANIE(params, self.ft)
        
    def get_param(self, shape):
        # a function to initialize embedding
        param = Parameter(torch.empty(shape, requires_grad=True, device=self.device))
        torch.nn.init.xavier_normal_(param.data)
        return param

    def add_base(self):
        if self.core == 'rgat':
            model = RGATLayerWrapper(None, None, self.num_e, self.num_rel, self.act, drop1=self.drop, drop2=self.drop,
                                       sub=None, rel=None, params=self.p)
        elif self.core == 'mgcn':
            model = MGCNLayerWrapper(None, None, self.num_e, self.num_rel, self.act, drop1=self.drop, drop2=self.drop,
                                       sub=None, rel=None, params=self.p)
        model.to(self.device)
        return model
    
    # def construct_gde_func(self):
    #     gdefunc = self.add_base()
    #     return gdefunc

    # def construct_GDEBlock(self, gdefunc):
    #     gde = ODEBlock(odefunc=gdefunc, method=self.solver, atol=self.atol, rtol=self.rtol, adjoint=self.adjoint_flag).to(self.device)
    #     return gde

    def TuckER(self):
        W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.hidsize, self.hidsize, self.hidsize)),
                                    dtype=torch.float, device=self.device, requires_grad=True))
        input_dropout = torch.nn.Dropout(self.drop)
        hidden_dropout1 = torch.nn.Dropout(self.drop)
        hidden_dropout2 = torch.nn.Dropout(self.drop)

        bn0 = torch.nn.BatchNorm1d(self.hidsize)
        bn1 = torch.nn.BatchNorm1d(self.hidsize)

        input_dropout.to(self.device)
        hidden_dropout1.to(self.device)
        hidden_dropout2.to(self.device)
        bn0.to(self.device)
        bn1.to(self.device)

        return W, input_dropout, hidden_dropout1, hidden_dropout2, bn0, bn1

    def Boy(self):
        W = torch.nn.Parameter(torch.empty((2*self.embsize+1,self.embsize), device=self.device, requires_grad=True))
        b = torch.nn.Parameter(torch.empty((1, self.embsize), device=self.device, requires_grad=True))
        torch.nn.init.uniform_(W.data, -1, 1)
        torch.nn.init.zeros_(b.data)
        return W, b
    
    def Jump(self):
        if self.p.rel_jump:
            jump = MGCNConvLayer(self.hidsize, self.hidsize, act=self.act, params=self.p, isjump=True, diag=True)
        else:
            jump = GCNConvLayer(self.hidsize, self.hidsize, act=self.act, params=self.p)

        jump.to(self.device)
        jump_weight = torch.FloatTensor([self.p.jump_init]).to(self.device)
        return jump, jump_weight

    def loss_comp(self, sub, rel, emb, label, tar_time, obj=None):
        score = self.score_comp(sub, rel, emb, tar_time)
        return self.loss(score, obj)


    def score_comp(self, sub, rel, emb, tar_time):
        sub_emb, rel_emb, all_emb = self.find_related(sub, rel, emb)
        if self.score_func.lower() == 'distmult':
            obj_emb = torch.cat([torch.index_select(self.emb_e, 0, sub), sub_emb], dim=1) * rel_emb.repeat(1,2)
            s = torch.mm(obj_emb, torch.cat([self.emb_e, all_emb], dim=1).transpose(1,0))

        elif self.score_func.lower() == 'tucker':
            x = self.bn0(sub_emb)
            x = self.input_dropout(x)
            x = x.view(-1, 1, sub_emb.size(1))

            W_mat = torch.mm(rel_emb, self.W_tk.view(rel_emb.size(1), -1))
            W_mat = W_mat.view(-1, sub_emb.size(1), sub_emb.size(1))
            W_mat = self.hidden_dropout1(W_mat)

            x = torch.bmm(x, W_mat)
            x = x.view(-1, sub_emb.size(1))
            x = self.bn1(x)
            x = self.hidden_dropout2(x)
            s = torch.mm(x, all_emb.transpose(1, 0))

        else:
            num = sub_emb.size(0)
            tar_time = tar_time.repeat(num,1).to(self.device)
            sub_emb, rel_emb, all_emb = self.find_related(sub, rel, emb)
            query = torch.tanh(torch.mm(torch.cat([sub_emb, rel_emb, tar_time],dim=1), self.W) + self.b)
            s = torch.mm(query, all_emb.transpose(1, 0))
        return s

    def find_related(self, sub, rel, emb):
        x = emb[:self.num_e,:]
        r = emb[self.num_e:,:]
        assert x.shape[0] == self.num_e
        assert r.shape[0] == self.num_rel * 2
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        return sub_emb, rel_emb, x

    def push_data(self, *args):
        out_args = []
        for arg in args:
            arg = [_arg.to(self.device) for _arg in arg]
            out_args.append(arg)
        return out_args

    def forward(self, sub_in, rel_in, obj_in, lab_in, sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list,
                edge_type_list, edge_id_jump, edge_w_jump, rel_jump):
        # self.test_flag = 0

        # push data onto gpu
        if self.p.jump:
            [sub_in, rel_in, obj_in, lab_in, sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump] = \
                        self.push_data(sub_in, rel_in, obj_in, lab_in, sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump)
        else:
            [sub_in, rel_in, obj_in, lab_in, sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list, edge_type_list] = \
                        self.push_data(sub_in, rel_in, obj_in, lab_in, sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list, edge_type_list)
        # for RE decoder
        # if self.score_func.lower() == 're' or self.score_func.lower():
        #     [obj_tar] = self.push_data(obj_tar)


        emb = torch.cat([self.emb_e, self.emb_r], dim=0)
        all_emb = torch.FloatTensor([]).to(self.device)
        for i in range(len(times)):
            self.graph_feature.set_graph(edge_index_list[i], edge_type_list[i])
               # ANIE
            if self.p.jump:
                if self.p.rel_jump:
                    self.graph_feature.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump,
                                                        jumpw=self.jump_weight,
                                                        skip=False, rel_jump=rel_jump[i])
                else:
                    self.graph_feature.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump, self.jump_weight,
                                                        False)
            dynimic_emb = self.graph_feature(times[i], emb)    
            # emb = self.odeblock.forward_nobatch(emb, start=times[i], end=tar_times[0], cheby_grid=self.p.cheby_grid)
            all_emb = torch.cat([all_emb, dynimic_emb], dim=0)

        all_emb = all_emb.view((self.num_e+2*self.num_rel), len(times), self.p.node_feats)
        times_stamps = times
        # all_emb = torch.cat([all_emb, emb], dim=0).view((self.num_e+2*self.num_rel), len(times)+1, self.p.node_feats)
        # times_stamps = torch.FloatTensor(times+tar_times) 
        
        final_emb = self.ANIEBlock(times_stamps, all_emb)
        loss = self.loss_comp(sub_tar[0], rel_tar[0], final_emb, lab_tar[0], tar_times[0], obj=obj_tar[0])

        return loss

    def forward_eval(self, times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump):
        # push data onto gpu
        if self.p.jump:
            [times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump] = \
                self.push_data(times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump)
        else:
            [times, tar_times, edge_index_list, edge_type_list, edge_index_list] = \
                self.push_data(times, tar_times, edge_index_list, edge_type_list, edge_index_list)

        emb = torch.cat([self.emb_e, self.emb_r], dim=0)
        all_emb = torch.FloatTensor([]).to(self.device)
        for i in range(len(times)):
            self.graph_feature.set_graph(edge_index_list[i], edge_type_list[i])
               # ANIE
            if self.p.jump:
                if self.p.rel_jump:
                    self.graph_feature.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump,
                                                        jumpw=self.jump_weight,
                                                        skip=False, rel_jump=rel_jump[i])
                else:
                    self.graph_feature.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump, self.jump_weight,
                                                        False)
            dynimic_emb = self.graph_feature(times[i], emb)    
            # emb = self.odeblock.forward_nobatch(emb, start=times[i], end=tar_times[0], cheby_grid=self.p.cheby_grid)
            all_emb = torch.cat([all_emb, dynimic_emb], dim=0)

        all_emb = all_emb.view((self.num_e+2*self.num_rel), len(times), self.p.node_feats)
        times_stamps = times
        # all_emb = torch.cat([all_emb, emb], dim=0).view((self.num_e+2*self.num_rel), len(times)+1, self.p.node_feats)
        # times_stamps = torch.FloatTensor(times+tar_times) 

        final_emb = self.ANIEBlock(times_stamps, all_emb)
        final_emb = emb
        return final_emb