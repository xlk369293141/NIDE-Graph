import torch
import torch.nn as nn
# import torchdiffeq
from source.solver import IDESolver, IDESolver_monoidal

num_MC = 30
class IDEBlock(nn.Module):
    def __init__(self, odefunc:nn.Module, kernel:nn.Module, F_func:nn.Module, method:str='dopri5', 
                 rtol:float=1e-3, atol:float=1e-4, adjoint:bool=True, ode_option=False):
        super(IDEBlock, self).__init__()
        self.odefunc = odefunc
        self.method = method
        self.atol, self.rtol = atol, rtol
        self.ode_option = ode_option
        self.adjoint_option = adjoint
        self.kernel = kernel
        self.F_func = F_func
        self.device = 'cuda:0'

        
    def forward(self, y_0:torch.Tensor, start:list, stop):
        if type(stop)==list:
            start = start + stop
        else:
            start.append(stop)
        times = torch.tensor(start).float()
        times = times.type_as(y_0)

        # out = self.odeint(self.odefunc, x, times,
        #                              rtol=self.rtol, atol=self.atol, method=self.method)
        alpha = lambda x:times[0].to(self.device)
        beta = lambda x:x.to(self.device)           
        out = IDESolver(times,
                                y_0,
                                c = self.odefunc,
                                k = self.kernel,
                                f = self.F_func,
                                lower_bound = alpha,
                                upper_bound = beta,
                                max_iterations = 3,
                                ode_option = self.ode_option,
                                adjoint_option = self.adjoint_option,
                                integration_dim = -2,
                                kernel_nn = True,
                                number_MC_samplings=num_MC,
                                ).solve()
        out = out.transpose(0,1)
        return out[-1]

    def forward_nobatch(self, y_0: torch.Tensor, start: float, end: float):
        times = torch.tensor([start,end]).float()
        times = times.type_as(y_0)

        # out = self.odeint(self.odefunc, x, times,
        #                              rtol=self.rtol, atol=self.atol, method=self.method)
        alpha = lambda x:times[0].to(self.device)
        beta = lambda x:x.to(self.device) 
        out = IDESolver(times,
                                y_0,
                                c = self.odefunc,
                                k = self.kernel,
                                f = self.F_func,
                                lower_bound = alpha,
                                upper_bound = beta,
                                max_iterations = 3,
                                ode_option = self.ode_option,
                                adjoint_option = self.adjoint_option,
                                integration_dim = -2,
                                kernel_nn = True,
                                number_MC_samplings=num_MC,
                                ).solve()
        out = out.transpose(0,1)
        return out[-1]

    def trajectory(self, y_0:torch.Tensor, T:int, num_points:int):
        times = torch.linspace(0, T, num_points)
        times = times.type_as(y_0)
        # out = self.odeint(self.odefunc, x, times,
        #                          rtol=self.rtol, atol=self.atol, method=self.method)
        alpha = lambda x:times[0].to(self.device)
        beta = lambda x:x.to(self.device) 
        out = IDESolver(times,
                                y_0,
                                c = self.odefunc,
                                k = self.kernel,
                                f = self.F_func,
                                lower_bound = alpha,
                                upper_bound = beta,
                                max_iterations = 3,
                                ode_option = self.ode_option,
                                adjoint_option = self.adjoint_option,
                                integration_dim = -2,
                                kernel_nn = True,
                                number_MC_samplings=num_MC,
                                ).solve()
        out = out.transpose(0,1)
        return out

