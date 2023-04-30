import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .odeblock import ODEBlock
from .MGCN import *
from .MGCNLayer import *
from .ideblock import IDEBlock
import source.kernels as Kernels
from source.ide_func import NNIDEF, NeuralIDE, NNIDEF_wODE, NeuralIDE_wODE

class NN_feedforward(nn.Module):
    def __init__(self, in_dim, hid_dim,out_dim):
        super(NN_feedforward, self).__init__()

        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)
        self.ELU = nn.ELU(inplace=True)

    def forward(self,y):
        y_in = y
        
        h = self.ELU(self.lin1(y_in))
        # h = self.ELU(self.lin2(h))
        out = self.lin2(h)
        
        return out
    
# class Simple_NN(nn.Module):
#     def __init__(self, in_dim, hid_dim,out_dim):
#         super(Simple_NN, self).__init__()

#         self.lin1 = nn.Linear(in_dim+1, hid_dim)
#         self.lin2 = nn.Linear(hid_dim, hid_dim)
#         self.lin3 = nn.Linear(hid_dim, hid_dim)
#         self.lin4 = nn.Linear(hid_dim, out_dim)
#         self.ELU = nn.ELU(inplace=True)
        
#         self.in_dim = in_dim

#     def forward(self,x,y):
#         y = y
#         x = x.view(1,1).repeat(y.shape[0],1)
        
#         y_in = torch.cat([x,y],-1)
#         h = self.ELU(self.lin1(y_in))
#         h = self.ELU(self.lin2(h))
#         h = self.ELU(self.lin3(h))
#         out = self.lin4(h)
        
#         return out

class StateHistory(nn.Module):
    def __init__(self, params, num_e, act):
        super().__init__()
        self.p = params
        self.num_e = num_e
        self.act = act
        self.net = MGCNConvLayer(self.p.hidsize, self.p.hidsize, act=self.act, params=self.p, isjump=True, diag=True)
        if self.p.res:
            self.res = torch.nn.Parameter(torch.FloatTensor([0.1]))
        self.drop = torch.nn.Dropout(self.p.dropout)
        
    def set_history(self, edge_id_his, edge_w_his, rel_his=None):
        self.edge_id_his = edge_id_his
        self.edge_w_his = edge_w_his
        self.rel_his = rel_his
  
    def forward(self, emb):
        tmp = self.res * self.net.forward(emb, self.edge_id_his, self.rel_his, self.num_e,
													dN=self.edge_w_his)
        emb = emb + tmp
        emb = self.drop(emb)
        return emb, tmp
    
class TANGO(nn.Module):
    def __init__(self, num_e, num_rel, params, device, logger):
        super().__init__()

        self.num_e = num_e
        self.num_rel = num_rel
        self.p = params
        self.core = self.p.gde_core
        self.core_layer = self.p.core_layer
        self.score_func = self.p.score_func
        self.solver = self.p.solver
        self.rtol = self.p.rtol
        self.atol = self.p.atol
        self.scale = self.p.scale
        self.device = self.p.device
        self.initsize = self.p.initsize
        self.adjoint_flag = self.p.adjoint_flag
        self.odefunc_flag = self.p.odefunc_flag
        self.drop = self.p.dropout
        self.hidsize = self.p.hidsize
        self.embsize = self.p.embsize
        self.his = self.p.his
        self.his_reg = self.p.his_reg
        self.his_count=0
        
        
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
        self.emb_e = self.get_param((self.num_e, self.initsize))    #20G
        self.emb_r = self.get_param((self.num_rel * 2, self.initsize))

        # define graph ode core
        self.gde_func = self.construct_gde_func()

        # define ode block
        if self.p.odefunc_flag:
            self.odeblock = ODEBlock(odefunc=self.gde_func, method=self.solver, atol=self.atol, rtol=self.rtol, adjoint=self.adjoint_flag).to(self.device)
        else:
            self.kernel, self.F_func = self.Integoral()
            self.ideblock = IDEBlock(odefunc=self.gde_func, kernel=self.kernel, F_func=self.F_func, method=self.solver, atol=self.atol, rtol=self.rtol, adjoint=self.adjoint_flag, ode_option=self.odefunc_flag).to(self.device)
            

        # define jump modules
        if self.p.jump:
            self.jump, self.jump_weight = self.Jump()
            self.gde_func.jump = self.jump
            self.gde_func.jump_weight = self.jump_weight

        # score function TuckER
        if self.score_func.lower() == "tucker":
            self.W_tk, self.input_dropout, self.hidden_dropout1, self.hidden_dropout2, self.bn0, self.bn1 = self.TuckER()

        if self.p.his:
            self.historyfunc = StateHistory(self.p, self.num_e, self.act)
            self.historyfunc.to(self.device)
        
    def get_param(self, shape):
        # a function to initialize embedding
        param = Parameter(torch.empty(shape, requires_grad=True, device=self.device))
        torch.nn.init.xavier_normal_(param.data)
        return param
    
    def Integoral(self):
        kernel = Kernels.kernel_NN_nbatch(self.embsize, self.embsize, [200,200])
        F_func = NN_feedforward(self.embsize,200,self.hidsize)
        # F_func = self.construct_gde_func()
        # int_jump, int_jump_weight = self.Jump()
        # F_func.jump = int_jump
        # F_func.jump_weight = int_jump_weight
        return kernel, F_func
        
    def add_base(self):
        model = MGCNLayerWrapper(None, None, self.num_e, self.num_rel, self.act, drop1=self.drop, drop2=self.drop,
                                       sub=None, rel=None, params=self.p)
        model.to(self.device)
        return model

    def construct_gde_func(self):
        gdefunc = self.add_base()
        return gdefunc

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

    def Jump(self):
        if self.p.rel_jump:
            jump = MGCNConvLayer(self.hidsize, self.hidsize, act=self.act, params=self.p, isjump=True, diag=True)
        else:
            jump = GCNConvLayer(self.hidsize, self.hidsize, act=self.act, params=self.p)

        jump.to(self.device)
        jump_weight = torch.FloatTensor([self.p.jump_init]).to(self.device)
        return jump, jump_weight

    def loss_comp(self, sub, rel, emb, label, obj=None):
        score = self.score_comp(sub, rel, emb)
        return self.loss(score, obj)


    def score_comp(self, sub, rel, emb):
        sub_emb, rel_emb, all_emb = self.find_related(sub, rel, emb)
        if self.score_func.lower() == 'distmult':
            obj_emb = torch.cat([torch.index_select(self.emb_e, 0, sub), sub_emb], dim=1) * rel_emb.repeat(1,2)
            s = torch.mm(obj_emb, torch.cat([self.emb_e, all_emb], dim=1).transpose(1,0))

        if self.score_func.lower() == 'tucker':
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
                edge_type_list, edge_id_jump, edge_w_jump, rel_jump, edge_id_his, edge_w_his, rel_his):
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
        loss = torch.FloatTensor([0.0]).to(self.device)

        if self.his:
            [edge_id_his] = self.push_data(edge_id_his)
            edge_w_his = edge_w_his.to(self.device)
            rel_his = rel_his.to(self.device)
            
            self.historyfunc.set_history(edge_id_his, edge_w_his, rel_his=rel_his)
            if len(self.historyfunc.edge_id_his) >0:
                emb, tmp = self.historyfunc(emb)
                loss += self.his_reg * (self.scale-times[0]) * torch.norm(tmp)
            else:
                self.his_count += 1
            # loss += self.loss_comp(sub_in[0], rel_in[0], emb, lab_in[0], obj = obj_in[0])
            
        if self.odefunc_flag:
            if self.p.jump:
                if self.p.rel_jump:
                    self.odeblock.odefunc.set_batch(times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, self.jump, self.jump_weight, rel_jump)
                else:
                    self.odeblock.odefunc.set_batch(times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, self.jump, self.jump_weight)
            else:
                self.odeblock.odefunc.set_batch(times, edge_index_list, edge_type_list)
            
            emb = self.odeblock.forward_nobatch(emb, start=times, end=tar_times[0])
        else:
            if self.p.jump:
                if self.p.rel_jump:
                    self.ideblock.odefunc.set_batch(times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, self.jump, self.jump_weight, rel_jump)
                    # self.ideblock.F_func.set_batch(times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, self.jump, self.jump_weight, rel_jump)
                else:
                    self.ideblock.odefunc.set_batch(times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, self.jump, self.jump_weight)
                    # self.ideblock.F_func.set_batch(times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, self.jump, self.jump_weight)
            else:
                self.ideblock.odefunc.set_batch(times, edge_index_list, edge_type_list)
                # self.ideblock.F_func.set_batch(times, edge_index_list, edge_type_list)
                
            emb = self.ideblock.forward(emb, start=times, stop=tar_times[0]) 

        loss += self.loss_comp(sub_tar[0], rel_tar[0], emb, lab_tar[0], obj=obj_tar[0])

        return loss

    def forward_eval(self, times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump, edge_id_his, edge_w_his, rel_his):
        # push data onto gpu
        if self.p.jump:
            [times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump] = \
                    self.push_data(times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump)
        else:
            [times, tar_times, edge_index_list, edge_type_list] = \
                        self.push_data(times, tar_times, edge_index_list, edge_type_list)

        emb = torch.cat([self.emb_e, self.emb_r], dim=0)

        if self.his:
            [edge_id_his] = self.push_data(edge_id_his)
            edge_w_his.to(self.device)
            rel_his.to(self.device)
            self.historyfunc.set_history(edge_id_his, edge_w_his, rel_his=rel_his)
            if len(self.historyfunc.edge_id_his):
                emb = self.historyfunc(emb)
            else:
                self.his_count += 1
        if self.odefunc_flag:
            if self.p.jump:
                if self.p.rel_jump:
                    self.odeblock.odefunc.set_batch(times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, self.jump, self.jump_weight, rel_jump)
                else:
                    self.odeblock.odefunc.set_batch(times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, self.jump, self.jump_weight)
            else:
                self.odeblock.odefunc.set_batch(times, edge_index_list, edge_type_list)
            emb = self.odeblock.forward_nobatch(emb, start=times, stop=tar_times[0])
        else:
            if self.p.jump:
                if self.p.rel_jump:
                    self.ideblock.odefunc.set_batch(times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, self.jump, self.jump_weight, rel_jump)
                    # self.ideblock.F_func.set_batch(times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, self.jump, self.jump_weight, rel_jump)
                else:
                    self.ideblock.odefunc.set_batch(times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, self.jump, self.jump_weight)
                    # self.ideblock.F_func.set_batch(times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, self.jump, self.jump_weight)
            else:
                self.ideblock.odefunc.set_batch(times, edge_index_list, edge_type_list)
                # self.ideblock.F_func.set_batch(times, edge_index_list, edge_type_list)
                
            emb = self.ideblock.forward(emb, start=times, stop=tar_times[0])

        return emb