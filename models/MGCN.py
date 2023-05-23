import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .MGCNLayer import MGCNConvLayer

class MGCNLayerWrapper(torch.nn.Module):
	def __init__(self, edge_index, edge_type, num_e, num_rel, act, drop1, drop2, sub, rel, params=None):
		super().__init__()
		self.nfe = 0
		self.p = params
		self.edge_index = edge_index
		self.edge_type = edge_type
		self.p.hidsize = self.p.embsize if self.p.core_layer == 1 else self.p.hidsize
		self.num_e = num_e
		self.num_rel = num_rel
		self.device = self.p.device
		self.act = act
		self.drop_l1 = torch.nn.Dropout(drop1)
		self.drop_l2 = torch.nn.Dropout(drop2)
		self.sub = sub
		self.rel = rel
		self.jump = None
		self.jump_weight = None
		self.max_idx = torch.tensor(self.p.input_step-1).int().cuda()
		# residual layer
		if self.p.res:
			self.res1 = torch.nn.Parameter(torch.FloatTensor([0.1]))
			self.res2 = torch.nn.Parameter(torch.FloatTensor([0.1]))
			if self.p.jump:
				self.jump_res = torch.nn.Parameter(torch.FloatTensor([0.1]))
				self.jump_drop = torch.nn.Dropout(drop2)
    
		# define MGCN Layer
		self.conv1 = MGCNConvLayer(self.p.initsize, self.p.hidsize, act=self.act, params=self.p)
		self.conv2 = MGCNConvLayer(self.p.hidsize, self.p.embsize, act=self.act, params=self.p) if self.p.core_layer == 2 else None

		# self.register_parameter('bias', Parameter(torch.zeros(num_e)))

	def set_graph(self, edge_index, edge_type):
		self.edge_index = edge_index
		self.edge_type = edge_type

	def set_jumpgraph(self, edge_id_jump, edge_w_jump, skip=False, rel_jump=None):
		self.edge_id_jump = edge_id_jump
		self.edge_w_jump = edge_w_jump
		# self.jump = jumpfunc
		# self.jump_weight = jumpw
		self.skip = skip
		self.rel_jump = rel_jump

	def set_batch(self, times, edge_index_list, edge_type_list, edge_id_jump_list=None, edge_w_jump_list=None, rel_jump_list=None):
		self.times = times
		self.edge_index_list = edge_index_list
		self.edge_type_list = edge_type_list
		self.edge_id_jump_list = edge_id_jump_list
		self.edge_w_jump_list = edge_w_jump_list
		self.rel_jump_list = rel_jump_list
  
	def forward(self, t, emb):
		times = torch.tensor(self.times).float().cuda()
		idx = (times<=t).sum(dim=-1)-1
		assert(idx>=0)
		idx = torch.min(idx, self.max_idx)
		self.set_graph(self.edge_index_list[idx], self.edge_type_list[idx])
  
		self.nfe += 1
		if self.p.res:
			emb = emb + self.res1 * self.conv1(emb, self.edge_index, self.edge_type, self.num_e)
			emb = self.drop_l1(emb)
			emb = (emb + self.res2 * self.conv2(emb, self.edge_index, self.edge_type, self.num_e)) if self.p.core_layer == 2 else emb
			emb = self.drop_l2(emb) if self.p.core_layer == 2 else emb

		else:
			emb	= self.conv1(emb, self.edge_index, self.edge_type, self.num_e)
			emb	= self.drop_l1(emb)
			emb	= self.conv2(emb, self.edge_index, self.edge_type, self.num_e) 	if self.p.core_layer == 2 else emb
			emb	= self.drop_l2(emb) 							if self.p.core_layer == 2 else emb
 
		return emb

	def forward_jump(self, t, emb):
		times = torch.tensor(self.times).float().cuda()
		idx_batch = (times<=t).sum(dim=-1)-1
		idx_batch = torch.min(idx_batch, self.max_idx)
		out = []
		emb=emb.squeeze()
		for batch_id, idx in enumerate(idx_batch):
			if self.p.rel_jump:
				self.set_jumpgraph(self.edge_id_jump_list[idx], self.edge_w_jump_list[idx],
										skip=False, rel_jump=self.rel_jump_list[idx])
			else:
				self.set_jumpgraph(self.edge_id_jump_list[idx], self.edge_w_jump_list[idx],
									skip=False)
			
			if self.p.rel_jump:
				jump_res = self.jump.forward(emb[:,batch_id,:], self.edge_id_jump, self.rel_jump, self.num_e,
													dN=self.edge_w_jump)
			else:
				jump_res = self.jump(emb[:,batch_id,:], self.edge_id_jump, dN=self.edge_w_jump)	
			emb_ = emb[:,batch_id,:] + self.jump_res * jump_res
			emb_ = self.jump_drop(emb_)
			out.append(emb_.unsqueeze(1))
		out = torch.cat(out,dim=1)
		return out
