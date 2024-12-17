# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from Logconfig import LoggerManager

logger = LoggerManager.get_logger()


class SSIR_ODEFIT(nn.Module):
    def __init__(self, mtype, obs_len, pre_len):
        super(SSIR_ODEFIT, self).__init__()
        self.mtype = mtype
        self.obs_len = obs_len
        self.pre_len = pre_len
        
        self.epidemic_module = EpidemicModule(mtype, obs_len)
        self.forecasting_module = ForecastingModule(mtype, pre_len)

    def forward(self, x0, params, forcast=False):
        # (B, 1, N, F), (B, T_obs, N, 2)  -> (B, T_obs, N, F)
        if forcast:
            x_pre, epiparams = self.forecasting_module(x0, params)
        else:
            x_pre, epiparams = self.epidemic_module(x0, params)
        
        return x_pre, epiparams


class EpidemicModule(nn.Module):
    def __init__(self, mtype, obs_len):
        super(EpidemicModule, self).__init__()
        self.mtype = mtype
        self.obs_len = obs_len

    def forward(self, x0, params):
        # (B, 1, N, F), (B, T_obs, N, 2)
        
        batch, node, feature = x0.size()
        
        betas = F.sigmoid(params[...,0:1]) # (B, T_obs, N, 1)
        gammas = F.sigmoid(params[...,1:2]) # (B, T_obs, N, 1)
        if self.mtype == 'ssir':
            cs = F.softmax(params[...,2:], dim=-1)
            
        x_init = x0 # (B, N, F)
        output = torch.zeros(
            batch,
            node,
            self.obs_len,
            feature, # S,I,R
            dtype=x_init.dtype,
            device=x_init.device,
        )
        
        for t in range(self.obs_len):
            if self.mtype == 'ssir':
                nsir = sircell(x_init, betas[:, t, :, :],gammas[:, t, :, :],cs[:, t, :, :])
            else:
                nsir = sircell(x_init, betas[:, t, :, :],gammas[:, t, :, :])
            x_init = nsir # (B, N, [S, I, R])
            output[:, :, t, :] = nsir # (B, N, obs_len, [S, I, R])
        
        output_view = output.permute(0, 2, 1, 3)  # (B, obs_len, N, 3)
        if self.mtype == 'ssir':
            epiparams = torch.cat([betas, gammas, cs], dim=-1)
        else:
            epiparams = torch.cat([betas, gammas], dim=-1)

        # SIR
        return output_view, epiparams
    
def sircell(sir, beta, gamma, c=None):
    S, I, R = sir[:, :, 0:1], sir[:, :, 1:2], sir[:, :, 2:]  # (B, N)
        
    epsilon = 1e-8  # Prevent division by zero
    N = (S + I + R).clamp(min=epsilon)  # (B, N, 1)
    if c is not None:
        infection_term = torch.matmul(c, I)  # (B, N, 1)
        delta_S = -beta * S / N * infection_term
    else:   
        delta_S = -beta * S * I / N
    delta_I = -delta_S - gamma * I
    delta_R = gamma * I
        
    # Ensure non-negativity
    S_t = torch.clamp(S + delta_S, min=0.0)
    I_t = torch.clamp(I + delta_I, min=0.0)
    R_t = torch.clamp(R + delta_R, min=0.0)
    
    if c is not None:
        # normalizaion
        total = S_t + I_t + R_t
        scaling_factor = N / total.clamp(min=epsilon)
        S_t = S_t * scaling_factor
        I_t = I_t * scaling_factor
        R_t = R_t * scaling_factor
        
    return torch.cat((S_t, I_t, R_t), dim=2)  # (B, N, 3)


class ForecastingModule(nn.Module):
    def __init__(self, mtype, pre_len):
        super(ForecastingModule, self).__init__()
        self.mtype = mtype
        self.pre_len = pre_len

    def forward(self, x0, params):
        # (B, 1, N, F), (B, T_obs, N, 2)
        batch, node, feature = x0.size()
        
        betas = params[...,0:1] # (B, T_obs, N, 1)
        gammas = params[...,1:2] # (B, T_obs, N, 1)        
        if self.mtype == 'ssir':
            cs = F.softmax(params[...,2:], dim=-1)
        else:
            cs = None
        
        x_init = x0 # (B, N, F)
        output = torch.zeros(
            batch,
            node,
            self.pre_len,
            feature, # S,I,R
            dtype=x_init.dtype,
            device=x_init.device,
        )
        
        for t in range(self.pre_len):
            if self.mtype == 'ssir':
                nsir = sircell(x_init, betas[:, t, :, :],gammas[:, t, :, :],cs[:, t, :, :])
            else:
                nsir = sircell(x_init, betas[:, t, :, :],gammas[:, t, :, :])
            x_init = nsir  # (B, N, [S, I, R])
            output[:, :, t, :] = nsir   # (B, N, obs_len, [S, I, R])
        
        output_view = output.permute(0, 2, 1, 3)  # (B, obs_len, N, 3)
        if self.mtype == 'ssir':
            epiparams = torch.cat([betas, gammas, cs], dim=-1)
        else:
            epiparams = torch.cat([betas, gammas], dim=-1)

        # SIR
        return output_view[:,:,:,1:2], epiparams
