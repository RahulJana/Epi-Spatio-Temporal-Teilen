# -*- coding: utf-8 -*-

import os
import torch

import numpy as np
import pandas as pd
from torch import nn, optim
from datetime import datetime
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler

import EpiGCN, EpiGAT, EpiODEfit, Toolkits, Constant
from Logconfig import LoggerManager

logger = LoggerManager.get_logger()


class Trainer(object):
    """
    Trainer class for training and testing epidemic models.
    
    Attributes:
        params (dict): Configuration parameters.
        model_type (str): Type of the model to train/test.
        adj_type (str): Type of adjacency matrix ('Static' or 'Dynamic').
        neighbor_matrix (torch.Tensor): Adjacency matrix tensor.
        model (nn.Module): The model instance.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        model_dir (str): Directory to save the model and logs.
    """
    def __init__(self, params: dict, model_type: str):
        self.params = params
        self.model_type = model_type

        # Validate required parameters
        required_params = {
            "graph_type", "data_type", "dev", "output_dir", "loss_type",
            "optimizer", "learning_rate", "weight_decay", "momentum",
            "scheduler", "step_size", "gamma", "milestones",
            "factor", "patience", "t_max", "eta_min", "early_stop",
            "grad_print", "clip", "obs_len", "pre_len", "kernel_size",
            "num_layers", "beta_incorporated", "max_horizon", "max_epoch",
            "normalize"
        }

        missing_params = required_params - self.params.keys()
        if missing_params:
            raise ValueError(f"Missing configuration parameters: {missing_params}")

        # Adjacency matrix type
        self.adj_type = self.params["graph_type"]

        if self.adj_type == 'Static':
            path = Constant.Paths.NEIGHBOR_ADJACENCY_MATRIX.format(data_type=self.params["data_type"])
            if not os.path.exists(path):
                raise FileNotFoundError(f"Neighbor matrix file not found at {path}")
            self.neighbor_matrix = torch.tensor(
                pd.read_csv(path, index_col=0).values, 
                dtype=torch.float32
            ).to(self.params['dev'])
            
        else:
            self.neighbor_matrix = None # never use later

        self.model = self.get_model().to(self.params["dev"])
        self.model.apply(self.init_weights)

        self.criterion = self.get_loss()
        if self.model_type != 'SSIR_ODEFIT': # When it's SSIR_ODEFIT, lazy initialization
            self.optimizer, self.scheduler = self.get_optimizer(self.model.parameters())

        self.model_dir = Toolkits.mkdirs(os.path.join(self.params["output_dir"], self.model_type))

    def init_weights(self, m):
        """
        Initialize model weights based on layer type.
        """
        if isinstance(m, nn.Linear):  
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # He initialization for ReLU
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
                    
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)  # Xavier initialization
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)  # Orthogonal initialization
                elif 'bias' in name:
                    param.data.fill_(0)  # Zero bias
                        
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)  # Scale initialized to 1
            torch.nn.init.zeros_(m.bias)  # Zero bias
                
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # He initialization
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def get_model(self) -> nn.Module:
        """
        Instantiate and return the appropriate model based on `model_type`.
        """
        if self.model_type == "SSIR_STGCN":
            model = EpiGCN.SSIR_STGCN(
                obs_len=self.params["obs_len"],
                pre_len=self.params["pre_len"],
                kernel_size=self.params["kernel_size"],
                num_layers=self.params["num_layers"],
                adj_type=self.adj_type,
                neighbor_matrix=self.neighbor_matrix,
                in_dim=3,
                t_out_dim=self.params.get("t_out_dim", 16),
                s_out_dim=self.params.get("s_out_dim", 16),
                dropout=self.params.get("dropout", 0.1),
                nsize=31 if (self.params['data_type'] == 'china') else 16, # 16 is for germany
                beta_incorporated=self.params['beta_incorporated'],
            )

        elif self.model_type == "SSIR_STGAT":
            model = EpiGAT.SSIR_STGAT(
                obs_len=self.params["obs_len"],
                pre_len=self.params["pre_len"],
                kernel_size=self.params["kernel_size"],
                num_layers=self.params["num_layers"],
                in_dim=3,
                nsize=31 if (self.params['data_type'] == 'china') else 16, # 16 is for germany
                t_out_dim=self.params.get("t_out_dim", 16),
                s_out_dim=self.params.get("s_out_dim", 16),
                dropout=self.params.get("dropout", 0.1),
                beta_incorporated=self.params['beta_incorporated'],
            )

        elif self.model_type == "SSIR_ODEFIT":
            model = EpiODEfit.SSIR_ODEFIT(
                mtype=self.params["ssir"],
                obs_len=self.params["obs_len"], 
                pre_len=self.params["pre_len"]
            )

        else:
            raise NotImplementedError(f"Unsupported model: {self.model_type}.")

        return model

    def get_loss(self) -> nn.Module:
        """
        Instantiate and return the appropriate loss function based on `loss_type`.
        """
        loss_type = self.params["loss_type"]
        if loss_type == "MSE":
            criterion = nn.MSELoss()
        elif loss_type == "cMSE":
            criterion = Toolkits.CustomMSELoss()
        elif loss_type == "MAE":
            criterion = nn.L1Loss()
        elif loss_type == "cMAE":
            criterion = Toolkits.CustomMAELoss()
        else:
            raise NotImplementedError(f"Unsupported loss function: {loss_type}.")
        
        return criterion

    def get_optimizer(self, parameters) -> tuple:
        """
        Instantiate and return the optimizer and scheduler based on configuration.
        """
        optimizer_type = self.params["optimizer"]
        learning_rate = self.params.get("learning_rate", 0.001)
        weight_decay = self.params.get("weight_decay", 0.0)
        momentum = self.params.get("momentum", 0.9)

        if optimizer_type == "Adam":
            optimizer = optim.Adam(
                params=parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "SGD":
            optimizer = optim.SGD(
                params=parameters,
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "RMSprop":
            optimizer = optim.RMSprop(
                params=parameters,
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            raise NotImplementedError(f"Unsupported optimizer: {optimizer_type}.")

        # Learning rate scheduler
        scheduler_type = self.params.get("scheduler", None)
        scheduler = None
        if scheduler_type == "StepLR":
            step_size = self.params.get("step_size", 10)
            gamma = self.params.get("gamma", 0.1)
            scheduler = lr_scheduler.StepLR(
                optimizer, 
                step_size=step_size, 
                gamma=gamma
            )
        elif scheduler_type == "MultiStepLR":
            milestones = self.params.get("milestones", [30, 80])
            gamma = self.params.get("gamma", 0.1)
            scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=gamma,
            )
        elif scheduler_type == "ExponentialLR":
            gamma = self.params.get("gamma", 0.95)
            scheduler = lr_scheduler.ExponentialLR(
                optimizer, 
                gamma=gamma
            )
        elif scheduler_type == "ReduceLROnPlateau":
            mode = self.params.get("mode", "min")
            factor = self.params.get("factor", 0.1)
            patience = self.params.get("patience", 10)
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode=mode, 
                factor=factor, 
                patience=patience
            )
        elif scheduler_type == "CosineAnnealingLR":
            T_max = self.params.get("t_max", 50)
            eta_min = self.params.get("eta_min", 0.0001)
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min,
            )
        elif scheduler_type is None:
            scheduler = None
        else:
            raise NotImplementedError(f"Unsupported scheduler: {scheduler_type}.")

        return optimizer, scheduler

    def train(self, data_loader: dict, modes: list):
        """
        Train the model using the provided data loader and modes.
        
        Args:
            data_loader (dict): Dictionary containing data loaders for different modes.
            modes (list): List of modes to train on (e.g., ["training", "validation"]).
        """
        logger.info(f"{self.model_type} model training begins.")

        start_time = datetime.now()

        max_horizon = self.params["max_horizon"]
        max_epoch = self.params["max_epoch"]
        current_horizon, epoch = 1, 0 

        losses = {mode: [] for mode in modes}
        losses_pre = {mode: [] for mode in modes}
        losses_phy = {mode: [] for mode in modes}
        losses_phy_y = {mode: [] for mode in modes}        

        lrs = {'epoch': [epoch], 'lr': [self.params['learning_rate']]}

        while current_horizon <= max_horizon:
            horizon_start_time = datetime.now()
            patience_count = self.params["early_stop"]        
            loss_all_threshold, loss_pre_threshold, loss_phy_threshold = np.inf, np.inf, np.inf
            current_epoch = 0

            while current_epoch <= max_epoch:
                epoch_start_time = datetime.now()

                epoch_loss = {mode: 0.0 for mode in modes}
                epoch_loss_pre = {mode: 0.0 for mode in modes}
                epoch_loss_phy = {mode: 0.0 for mode in modes}
                epoch_loss_phy_y = {mode: 0.0 for mode in modes}
            
                for mode in modes:
                    if mode == "training":
                        self.model.train()
                    else:
                        self.model.eval()

                    for batch_idx, (x_SIR, yd, yi) in enumerate(data_loader[mode]):
                        y = yd if self.params['daily'] else yi

                        x_SIR_obs = x_SIR[:, :self.params["obs_len"], :, :]  # (B, T_obs, N, F)
                        x_SIR_gd = x_SIR[:, self.params["obs_len"] : (self.params["obs_len"] + self.params["pre_len"]), :, :]  # (B, T_next, N, F)
                    
                        y_gd = y[:, self.params["obs_len"] : (self.params["obs_len"] + self.params["pre_len"]), :, :]  # (B, T_next, N, F)

                        with torch.set_grad_enabled(mode == "training"):
                            if self.model_type == "SSIR_STGCN":
                                if self.adj_type == "Static":
                                    y_pre, y_phy_i, epiparams, y_phy_all, y_phy_d, adj = self.model(x_SIR_obs)
                                elif self.adj_type == "Dynamic":
                                    y_pre, y_phy_i, epiparams, y_phy_all, y_phy_d, adj = self.model(x_SIR_obs)
                                elif self.adj_type == "Adaptive":
                                    y_pre, y_phy_i, epiparams, y_phy_all, y_phy_d, adj = self.model(x_SIR_obs)
                                else:
                                    raise NotImplementedError(f"Unsupported graph type: {self.params['graph_type']}.")

                            elif self.model_type == "SSIR_STGAT":
                                y_pre, y_phy_i, epiparams, y_phy_all, y_phy_d, adj = self.model(x_SIR_obs)
                            
                            elif self.model_type == "SSIR_ODEFIT":
                                y_pre, y_phy_i, epiparams, y_phy_all, y_phy_d, adj = self.model(x_SIR_obs)

                            else:
                                raise NotImplementedError(f"Invalid model: {self.model_type}.")

                            # Calculate losses
                            eloss_pre = self.criterion(y_pre[:, :current_horizon, :, :], y_gd[:, :current_horizon, :, :])
                            eloss_phy = self.criterion(y_phy_all[:, :current_horizon, :, :], x_SIR_gd[:, :current_horizon, :, :])
                            
                            if self.params['daily']:
                                eloss_phy_y = self.criterion(y_phy_d[:, :current_horizon, :, :], y_gd[:, :current_horizon, :, :])
                            else:
                                eloss_phy_y = self.criterion(y_phy_i[:, :current_horizon, :, :], y_gd[:, :current_horizon, :, :])
                                
                            if self.params['phyloss4all']:
                                eloss = self.params['w4pre']*eloss_pre + self.params['w4phy']*eloss_phy
                            else:
                                eloss = self.params['w4pre']*eloss_pre + self.params['w4phy']*eloss_phy_y
                                
                            if mode == "training":
                                self.optimizer.zero_grad()
                                eloss.backward()
                                
                                if self.params['grad_print']:
                                    Toolkits.check_gradients(self.model.named_parameters())

                                # Gradient clipping to prevent exploding gradients
                                if self.params["clip"] is not None:
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params["clip"])
                                self.optimizer.step()

                        # Accumulate loss for the epoch
                        epoch_loss[mode] += eloss.cpu().detach().item()
                        epoch_loss_pre[mode] += eloss_pre.cpu().detach().item()
                        epoch_loss_phy[mode] += eloss_phy.cpu().detach().item()
                        epoch_loss_phy_y[mode] += eloss_phy_y.cpu().detach().item()

                    # Save loss of each epoch
                    losses[mode].append(epoch_loss[mode])
                    losses_pre[mode].append(epoch_loss_pre[mode])
                    losses_phy[mode].append(epoch_loss_phy[mode])
                    losses_phy_y[mode].append(epoch_loss_phy_y[mode])

                    if mode != 'validation':
                        logger.info(
                            f"Epoch {epoch + current_epoch}, Horizon [{current_horizon}], Mode [{mode}]: "
                            f"Loss = {epoch_loss[mode]:.6f}, Pre Loss = {epoch_loss_pre[mode]:.6f}, "
                            f"Phy Loss = {epoch_loss_phy[mode]:.6f}, Phy_y Loss = {epoch_loss_phy_y[mode]:.6f}."
                        )

                    # Validation and early stopping
                    if mode == "validation":
                        epochloss = epoch_loss[mode]
                        epochloss_pre = epoch_loss_pre[mode]
                        epochloss_phy = epoch_loss_phy[mode]
                        epochloss_phy_y = epoch_loss_phy_y[mode]
                        
                        if self.params['phyloss4all']:
                            lossphy = epochloss_phy
                        else:
                            lossphy = epochloss_phy_y
                        
                        lr_before = self.optimizer.param_groups[0]['lr']
                        # Update learning rate scheduler
                        if self.params['scheduler'] == 'ReduceLROnPlateau':
                            self.scheduler.step(epochloss)  # Typically based on validation loss
                        else:
                            self.scheduler.step()  # Step-based schedulers
                        lr_after = self.optimizer.param_groups[0]['lr']
                        
                        if lr_after != lr_before:
                            lrs['epoch'].append(epoch + current_epoch)
                            lrs['lr'].append(lr_after)
                            
                        # Check for improvement
                        # if (epochloss_pre < loss_pre_threshold) and (lossphy < loss_phy_threshold):
                        if (self.params['w4phy'] == 1 and ((epochloss_pre < loss_pre_threshold) and (lossphy < loss_phy_threshold))) or (self.params['w4phy'] == 0 and (epochloss_pre < loss_pre_threshold)):
                            logger.info(
                                f"Epoch {epoch + current_epoch}, Horizon [{current_horizon}], Mode [{mode}]: "
                                f"Loss decreased from [{loss_all_threshold:.6f}, {loss_pre_threshold:.6f}, {loss_phy_threshold:.6f}] "
                                f"to [{epochloss:.6f}, {epochloss_pre:.6f}, {epochloss_phy:.6f}], Phy_y Loss: {epochloss_phy_y:.6f}. "
                                f"Saving model checkpoint."
                            )
                            
                            loss_all_threshold = epochloss
                            loss_pre_threshold = epochloss_pre
                            loss_phy_threshold = lossphy

                            checkpoint = {
                                "epoch": (epoch + current_epoch),
                                "state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                            }
                            torch.save(checkpoint, os.path.join(self.model_dir, f"{self.model_type}_optmodel.pkl"))

                            # Reset patience count
                            patience_count = self.params["early_stop"]

                        else:
                            logger.info(
                                f"Epoch {epoch + current_epoch}, Horizon [{current_horizon}], Mode [{mode}]: "
                                f"Loss did not improve from [{loss_all_threshold:.6f}, {loss_pre_threshold:.6f}, {loss_phy_threshold:.6f}], "
                                f"Phy_y Loss: {epochloss_phy_y:.6f}. Patience remaining: {patience_count - 1}."
                            )
                            
                            patience_count -= 1
                            # Early stopping condition
                            if patience_count == 0:
                                logger.warning(
                                    f"Early stopping at Epoch {epoch + current_epoch}, Horizon [{current_horizon}]. "
                                    f"Validation loss did not decrease after {self.params['early_stop']} epochs. "
                                    f"Time elapsed: {Toolkits.elapsed_time(horizon_start_time)}."
                                )
                                break  # Exit current epoch loop

                # Check if early stopping was triggered
                if patience_count == 0:
                    break  # Exit horizon loop

                logger.info(
                    f"Epoch {epoch + current_epoch}, Current Epoch {current_epoch}, "
                    f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}, "
                    f"Time Elapsed: {Toolkits.elapsed_time(epoch_start_time)}."
                )
                current_epoch += 1

            logger.info(
                f"Horizon [{current_horizon}] training completed. "
                f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}, "
                f"Time Elapsed: {Toolkits.elapsed_time(horizon_start_time)}."
            )
            current_horizon += 1 
            epoch += current_epoch

        # Plotting after training
        Toolkits.plot_lrcurve(lrs, self.model_dir)
        Toolkits.plot_losscurve_separately([losses, losses_pre, losses_phy, losses_phy_y], self.model_dir)

        logger.info(f"{self.model_type} model training ended. Total Time Elapsed: {Toolkits.elapsed_time(start_time)}.")
        
        pass

    def test(self, data_loader: dict, modes: list, auxdata):
        logger.info(f"{self.model_type} model testing begins.")

        # load trained model checkpoint
        trained_checkpoint = torch.load(os.path.join(self.model_dir, f"{self.model_type}_optmodel.pkl"), weights_only=True, map_location=torch.device(self.params["dev"]))

        # load model weight
        self.model.load_state_dict(trained_checkpoint["state_dict"])
        self.model.eval()

        errors = []
        for mode in modes:
            logger.info(f"{self.model_type} model testing on [{mode}] data begins.")

            x_obs_list, x_gd_list, y_obs_list, y_gd_list, y_pre_list, y_phy_list, y_epiparams_list, dadj_list = [], [], [], [], [], [], [], []
            with torch.no_grad():
                for x_SIR, yd, yi in data_loader[mode]:
                    y = yd if self.params['daily'] else yi
                    
                    x_SIR_obs = x_SIR[:, : self.params["obs_len"], :, :]  # (B,T_obs,N,F)
                    x_SIR_gd = x_SIR[:, self.params["obs_len"] : (self.params["obs_len"] + self.params["pre_len"]), :, :]  # (B,T_next,N,F)
                    
                    y_obs = y[:, : self.params["obs_len"], :, :]  # (B,T_obs,N,F)
                    y_gd = y[:, self.params["obs_len"] : (self.params["obs_len"] + self.params["pre_len"]), :, :]  # (B,T_next,N,F)

                    if self.model_type == "SSIR_STGCN":
                        if self.adj_type == "Static":
                            y_pre, y_phy_i, y_epiparams, _, y_phy_d, adj = self.model(x_SIR_obs)
                        elif self.adj_type == "Dynamic":
                            y_pre, y_phy_i, y_epiparams, _, y_phy_d, adj = self.model(x_SIR_obs)
                        elif self.adj_type == "Adaptive":
                            y_pre, y_phy_i, y_epiparams, _, y_phy_d, adj = self.model(x_SIR_obs)
                        else:
                            raise NotImplementedError(f"Unsupported graph type: {self.params['graph_type']}.")

                    elif self.model_type == "SSIR_STGAT":
                        y_pre, y_phy_i, y_epiparams, _, y_phy_d, adj = self.model(x_SIR_obs)

                    else:
                        raise NotImplementedError(f"Invalid model: {self.model_type}.")

                    x_obs_list.append(x_SIR_obs)
                    x_gd_list.append(x_SIR_gd)
                    y_obs_list.append(y_obs)
                    y_gd_list.append(y_gd)
                    y_pre_list.append(y_pre)
                    if self.params['daily']:
                        y_phy_list.append(y_phy_d)
                    else:
                        y_phy_list.append(y_phy_i)
                    y_epiparams_list.append(y_epiparams)
                    if (self.model_type == 'SSIR_STGCN' and self.adj_type == 'Dynamic') and adj is not None:
                        dadj_list.append(adj)

            # Concatenate all batches
            x_observation = torch.cat(x_obs_list, dim=0)
            x_ground_truth = torch.cat(x_gd_list, dim=0)
            y_ground_truth = torch.cat(y_gd_list, dim=0)
            y_prediction = torch.cat(y_pre_list, dim=0)
            y_physic = torch.cat(y_phy_list, dim=0)
            y_observation = torch.cat(y_obs_list, dim=0)
            y_epiparams_est = torch.cat(y_epiparams_list, dim=0)
            
            if self.params['normalize'] is not None:
                prov_pop = torch.tensor(auxdata['prov_pop'], dtype=y_ground_truth.dtype, device=y_ground_truth.device).view(1, 1, -1, 1)
                I_max = torch.tensor(auxdata['imax'], dtype=y_ground_truth.dtype, device=y_ground_truth.device).view(1, 1, -1, 1)
                I_min = torch.tensor(auxdata['imin'], dtype=y_ground_truth.dtype, device=y_ground_truth.device).view(1, 1, -1, 1)
                
                x_observation = x_observation * (I_max - I_min) + I_min
                x_ground_truth = x_ground_truth * (I_max - I_min) + I_min
                y_ground_truth = y_ground_truth * (I_max - I_min) + I_min
                y_prediction = y_prediction * (I_max - I_min) + I_min
                y_physic = y_physic * (I_max - I_min) + I_min
                y_observation = y_observation * (I_max - I_min) + I_min
                
                x_observation = x_observation * prov_pop
                x_ground_truth = x_ground_truth * prov_pop
                y_ground_truth = y_ground_truth * prov_pop
                y_prediction = y_prediction * prov_pop
                y_physic = y_physic * prov_pop
                y_observation = y_observation * prov_pop
            
            # errors
            errors.append([mode, 'Prediction'] + self.metrics(y_ground_truth, y_prediction))
            errors.append([mode, 'Physic'] + self.metrics(y_ground_truth, y_physic))
                
            # save to local
            Toolkits.save_foredata(self.model_dir, mode, ['x_obs.npy','x_gd.npy', 'y_obs.npy','y_gd.npy','y_pre.npy','y_phy.npy','y_epiparams.npy'], [x_observation, x_ground_truth, y_observation, y_ground_truth, y_prediction, y_physic, y_epiparams_est])
              
            logger.info(f"{self.model_type} model testing on [{mode}] end.")
                        
        # save errors to csv file
        Toolkits.save_foreerror(self.model_dir, errors, headers=['Mode', 'Type', "MAE", "cMAE", "MSE", "cMSE", "RMSE", "MAPE", "RAE", "PCC", "CCC"], filename=f"{self.model_type}_{self.params['data_type']}_{self.params['obs_len']}_{self.params['pre_len']}_{self.adj_type}_{self.params['w4phy']}_{self.params['ssir']}_errors.md")

        logger.info(f"{self.model_type} model testing end.")
        
        pass

    def metrics(self, ground_truth, forecast):
        # evaluate on metrics
        return self.evaluate(ground_truth, forecast)

    def evaluate(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        def MSE(y_pred: torch.Tensor, y_true: torch.Tensor):
            return F.mse_loss(y_pred, y_true)
        
        def cMSE(y_pred: torch.Tensor, y_true: torch.Tensor):
            return Toolkits.CustomMSELoss()(y_pred, y_true)

        def RMSE(y_pred: torch.Tensor, y_true: torch.Tensor):
            return torch.sqrt(MSE(y_pred, y_true))
        
        def MAE(y_pred: torch.Tensor, y_true: torch.Tensor):
            return F.l1_loss(y_pred, y_true)
        
        def cMAE(y_pred: torch.Tensor, y_true: torch.Tensor):
            return Toolkits.CustomMAELoss()(y_pred, y_true)

        def MAPE(y_pred: torch.Tensor, y_true: torch.Tensor):  # Avoid zero division
            epsilon = 1e-8  # Small constant to avoid division by zero
            loss = torch.abs((y_pred - y_true) / (y_true + epsilon))
            return torch.mean(loss)

        def RAE(y_pred: torch.Tensor, y_true: torch.Tensor):
            num = torch.sum(torch.abs(y_pred - y_true))
            denom = torch.sum(torch.abs(y_true - torch.mean(y_true)))
            epsilon = 1e-8  # Small constant to avoid division by zero
            return num / (denom + epsilon)

        def PCC(y_pred: torch.Tensor, y_true: torch.Tensor):
            mean_y_true = torch.mean(y_true)
            mean_y_pred = torch.mean(y_pred)
            cov = torch.mean((y_true - mean_y_true) * (y_pred - mean_y_pred))
            std_y_true = torch.std(y_true)
            std_y_pred = torch.std(y_pred)
            epsilon = 1e-8  # Small constant to avoid division by zero
            return cov / ((std_y_true * std_y_pred) + epsilon)
       
        def CCC(y_pred: torch.Tensor, y_true: torch.Tensor):
            mean_y_true = torch.mean(y_true)
            mean_y_pred = torch.mean(y_pred)
            var_y_true = torch.var(y_true)
            var_y_pred = torch.var(y_pred)
            cov = torch.mean((y_true - mean_y_true) * (y_pred - mean_y_pred))
            epsilon = 1e-8  # Small constant to avoid division by zero
            numerator = 2 * cov
            denominator = var_y_true + var_y_pred + (mean_y_true - mean_y_pred) ** 2 + epsilon
            return numerator / denominator

        mse = MSE(y_pred, y_true).item()
        cmse = cMSE(y_pred, y_true).item()
        rmse = RMSE(y_pred, y_true).item()
        mae = MAE(y_pred, y_true).item()
        cmae = cMAE(y_pred, y_true).item()
        mape = MAPE(y_pred, y_true).item()
        rae = RAE(y_pred, y_true).item()
        pcc = PCC(y_pred, y_true).item()
        ccc = CCC(y_pred, y_true).item()

        # logger.info(f"MAE: {mae}, cMAE: {cmae}, MSE: {mse}, cMSE: {cmse}, RMSE: {rmse}, MAPE: {mape}, RAE: {rae}, PCC: {pcc}, CCC: {ccc}.")

        return [mae, cmae, mse, cmse, rmse, mape, rae, pcc, ccc]
    
    def ode_estimator(self, data_loader: dict, auxdata, mode='SSIR_ODEfit'):
        """
            Estimate the epi-parameters of SIR model using the provided data loader.
        """
        logger.info(f"{self.model_type} model training begins.")
        start_time = datetime.now()
        
        x_SIR, yd, yi = Toolkits.rebuilddata(data_loader)
        y = yd if self.params['daily'] else yi
        
        # for odefit
        x_SIR_obs = x_SIR[:, :self.params["obs_len"], :, :]  # (B, T_obs, N, F)
        # for prediction
        # x_SIR_gd = x_SIR[:, self.params["obs_len"] : (self.params["obs_len"] + self.params["pre_len"]), :, :]  # (B, T_next, N, F)
        y_gd = y[:, self.params["obs_len"] : (self.params["obs_len"] + self.params["pre_len"]), :, :]  # (B, T_next, N, F)

        max_epoch = self.params["max_epoch"]
        patience_count = self.params["early_stop"]
        losses, loss_threshold = [], np.inf
        lrs = {'epoch': [], 'lr': []}
        
        batch, time, node, _ = x_SIR_obs.size()
        num_params = (node+2) if (self.params['ssir']=='ssir') else 2
        x_SIR_obs_params = nn.Parameter(torch.randn(batch, time, node, num_params, dtype=x_SIR_obs.dtype, device=x_SIR_obs.device), requires_grad=True) # (batch, T_obs, node, [beta, gamma])
        self.optimizer, self.scheduler = self.get_optimizer([x_SIR_obs_params])
        x_SIR_obs_initial = x_SIR_obs[:,0,:,:] # (batch, 1, node, feature)
        
        for epoch in range(max_epoch):
            epoch_start_time = datetime.now()
                    
            # for odefit
            with torch.set_grad_enabled(mode=True):
                # (B, N, F), (B, T, N, [epi-params]]) -> (B, T, N, [S,I,R]), (B, T, N, [epi-params])
                x_SIR_pre, learned_epiparams = self.model(x_SIR_obs_initial, x_SIR_obs_params)

                # Calculate losses
                loss = self.criterion(x_SIR_obs, x_SIR_pre)

                self.optimizer.zero_grad()
                loss.backward()
                                    
                if self.params['grad_print']:
                    Toolkits.check_gradients([['epiparams',x_SIR_obs_params]]) # [[]]

                # Gradient clipping to prevent exploding gradients
                if self.params["clip"] is not None:
                    torch.nn.utils.clip_grad_norm_(x_SIR_obs_params, self.params["clip"])
                self.optimizer.step()

                # Save loss of each epoch
                losses.append(loss.cpu().detach())

                # early stopping
                lr_before = self.optimizer.param_groups[0]['lr']
                # Update learning rate scheduler
                if self.params['scheduler'] == 'ReduceLROnPlateau':
                    self.scheduler.step(loss)  # Typically based on validation loss
                else:
                    self.scheduler.step()  # Step-based schedulers
                lr_after = self.optimizer.param_groups[0]['lr']
                        
                if lr_after != lr_before:
                    lrs['epoch'].append(epoch)
                    lrs['lr'].append(lr_after)
                            
                # Check for improvement
                if (loss < loss_threshold):
                    logger.info(f"Epoch {epoch}, Loss decreased from {loss_threshold:.6f} to {loss:.6f}, epiparams: {torch.equal(x_SIR_obs_params, learned_epiparams)}.")
                            
                    loss_threshold = loss
                    # Reset patience count
                    patience_count = self.params["early_stop"]

                    # here, can save the learned parameters to local    
                    torch.save(learned_epiparams, os.path.join(self.model_dir, f"{self.model_type}_optparams.pt"))
                else:
                    logger.info(f"Epoch {epoch}, Loss did not improve from [{loss_threshold:.6f}, Patience remaining: {patience_count - 1}.")
                            
                    patience_count -= 1
                    # Early stopping condition
                    if patience_count == 0:
                        logger.warning(f"Early stopping at Epoch {epoch}, loss did not decrease after {self.params['early_stop']} epochs.")
    
                        break

            logger.info(f"Epoch {epoch}, Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}, Time Elapsed: {Toolkits.elapsed_time(epoch_start_time)}.")

        # Plotting after training
        Toolkits.plot_lrcurve(lrs, self.model_dir)
        Toolkits.plot_losscurve(losses, self.model_dir, self.params["early_stop"])

        logger.info(f"{self.model_type} model training ended. Total Time Elapsed: {Toolkits.elapsed_time(start_time)}.")
        
        # forcasting
        x_SIR_obs_last = x_SIR_obs[:,-1,:,:]
        learned_params = torch.load(os.path.join(self.model_dir, f"{self.model_type}_optparams.pt"), weights_only=True, map_location=torch.device(self.params["dev"]))
        learned_params_repeat = learned_params.repeat_interleave(int(self.params['pre_len']/self.params['obs_len']), dim=1)
        
        errors = []
        with torch.set_grad_enabled(mode=False):
            # (B,N,F), (B,T,N,[2]) -> (B,T,B,[1]), _
            y_pre, epiparams = self.model(x_SIR_obs_last, learned_params_repeat, forcast=True)
            
            if self.params['normalize'] is not None:
                prov_pop = torch.tensor(auxdata['prov_pop'], dtype=y_gd.dtype, device=y_gd.device).view(1, 1, -1, 1)
                I_max = torch.tensor(auxdata['imax'], dtype=y_gd.dtype, device=y_gd.device).view(1, 1, -1, 1)
                I_min = torch.tensor(auxdata['imin'], dtype=y_gd.dtype, device=y_gd.device).view(1, 1, -1, 1)
                
                y_gd = y_gd * (I_max - I_min) + I_min
                y_pre = y_pre * (I_max - I_min) + I_min
                
                y_gd = y_gd * prov_pop
                y_pre = y_pre * prov_pop
            
            # errors
            errors.append([mode, 'Prediction'] + self.metrics(y_gd, y_pre))
 
            # save to local
            Toolkits.save_foredata(self.model_dir, mode, ['y_gd.npy', 'y_pre.npy'], [y_gd, y_pre])
            
            logger.info(f"{self.model_type} model testing on [{mode}] end, leanred epiparams: {torch.equal(learned_params_repeat, epiparams)}.")
                        
        # save errors to csv file
        Toolkits.save_foreerror(self.model_dir, errors, headers=['Mode', 'Type', "MAE", "cMAE", "MSE", "cMSE", "RMSE", "MAPE", "RAE", "PCC", "CCC"], filename=f"{self.model_type}_{self.params['data_type']}_{self.params['obs_len']}_{self.params['pre_len']}_{self.adj_type}_{self.params['w4phy']}_{self.params['ssir']}_errors.md")

        logger.info(f"{self.model_type} model testing end.")
            
        pass
