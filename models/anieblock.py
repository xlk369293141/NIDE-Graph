import torch
import torch.nn as nn
from IE_source.Attentional_IE_solver import Integral_attention_solver_multbatch


class ANIE(nn.Module):
    def __init__(self, params, Encoder):
        """ Standard ODEBlock class. Can handle all types of ODE functions
            :method:str = {'euler', 'rk4', 'dopri5', 'adams'}
        """
        super().__init__()
        self.p = params
        self.ft = Encoder

    def forward(self, time_stamps, rgat_nodes):
        z = Integral_attention_solver_multbatch(
                                        time_stamps,
                                        rgat_nodes[:,-1,:],
                                        y_init = rgat_nodes,
                                        c = None,
                                        Encoder = self.ft,
                                        lower_bound = lambda x: torch.Tensor([0]).to(self.p.device),
                                        upper_bound = lambda x: x,
                                        mask=None,
                                        sampling_points = time_stamps.size(0),
                                        use_support = False,
                                        n_support_points = 100,
                                        global_error_tolerance = 1e-5,
                                        max_iterations = self.p.max_iterations,
                                        # int_atol = self.p.atol,
                                        # int_rtol = self.p.rtol,
                                        smoothing_factor=self.p.smoothing_factor,
                                        store_intermediate_y = False,
                                        output_support_tensors=False,
                                        return_function=False,
                                        accumulate_grads=self.p.accumulate_grads,
                                        initialization=True                                       
                                        ).solve()
            
        return z[:,-1,:]

    def forward_nobatch(self, time_stamps, rgat_nodes):
        z = Integral_attention_solver_multbatch(
                                        time_stamps,
                                        rgat_nodes[:,-1,:],
                                        y_init = rgat_nodes,
                                        c = None,
                                        Encoder = self.ft,
                                        lower_bound = lambda x: torch.Tensor([0]).to(self.p.device),
                                        upper_bound = lambda x: x,
                                        mask=None,
                                        sampling_points = time_stamps.size(0),
                                        use_support = False,
                                        n_support_points = 100,
                                        global_error_tolerance = 1e-5,
                                        max_iterations = self.p.max_iterations,
                                        # int_atol = self.p.atol,
                                        # int_rtol = self.p.rtol,
                                        smoothing_factor=self.p.smoothing_factor,
                                        store_intermediate_y = False,
                                        output_support_tensors=False,
                                        return_function=False,
                                        accumulate_grads=self.p.accumulate_grads,
                                        initialization=True                                       
                                        ).solve()

        return z[:,-1,:]

    def trajectory(self, time_stamps, rgat_nodes):
        z = Integral_attention_solver_multbatch(
                                        time_stamps,
                                        rgat_nodes[:,-1,:],
                                        y_init = rgat_nodes,
                                        c = None,
                                        Encoder = self.ft,
                                        lower_bound = lambda x: torch.Tensor([0]).to(self.p.device),
                                        upper_bound = lambda x: x,
                                        mask=None,
                                        sampling_points = time_stamps.size(0),
                                        use_support = False,
                                        n_support_points = 100,
                                        global_error_tolerance = 1e-5,
                                        max_iterations = self.p.max_iterations,
                                        # int_atol = self.p.atol,
                                        # int_rtol = self.p.rtol,
                                        smoothing_factor=self.p.smoothing_factor,
                                        store_intermediate_y = False,
                                        output_support_tensors=False,
                                        return_function=False,
                                        accumulate_grads=self.p.accumulate_grads,
                                        initialization=True,                                      
                                        ).solve()
        return z[:,-1,:]