# -*- coding: utf-8 -*-

import os

import torch
import numpy as np
import pandas as pd

from tabulate import tabulate

from datetime import datetime
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.gridspec import GridSpec

from mpl_toolkits.axes_grid1 import make_axes_locatable

from Logconfig import LoggerManager


logger = LoggerManager.get_logger()

def rebuilddata(data_loader: dict):
    
    x_SIR_list, yd_list, yi_list = [], [], []
    for x_SIR, yd, yi in data_loader:
        x_SIR_list.append(x_SIR)
        yd_list.append(yd)
        yi_list.append(yi)
    
    x_SIR = torch.cat(x_SIR_list, dim=0)
    yd = torch.cat(yd_list, dim=0)
    yi = torch.cat(yi_list, dim=0)
    
    return x_SIR, yd, yi


def mkdirs(path):
    os.makedirs(path, exist_ok=True)
    
    return path
    
    
def CustomMSELoss(cdim=[0,1,3]):
    def loss_fn(y_pred, y_true):
        loss = (y_pred - y_true) ** 2
        loss = loss.mean(dim=cdim)
        return loss.sum()

    return loss_fn

# (B,T,N,F) 只在N这一维做了平均
def CustomMAELoss(cdim=[0,1,3]):
    def loss_fn(y_pred, y_true):
        loss = torch.abs(y_pred - y_true)
        loss = loss.mean(dim=cdim)
        return loss.sum()

    return loss_fn


def elapsed_time(start_time):
    end_time = datetime.now()

    elapsed_time = end_time - start_time
    elapsed_seconds = elapsed_time.total_seconds()

    seconds = int(elapsed_seconds)
    milliseconds = int((elapsed_seconds - seconds) * 1000)

    return f"{seconds}s.{milliseconds:04d}ms"


def plot_forecurve(path, datas, mode, horizon, province_names, time_ratio, offset):
    date_points = DATE.DATE_LIST[(time_ratio[0]+time_ratio[1]+offset[0]) : (time_ratio[0]+time_ratio[1]+time_ratio[2]-offset[1]+1)]
    
    if len(datas)>2:
        obs_value, pre_value, phy_value, mix_value = [data.cpu().detach().numpy() for data in datas]
    else:
        obs_value, pre_value = [data.cpu().detach().numpy() for data in datas]
    
    num_provinces = len(province_names)
    num_columns = 3 if len(datas)>2 else 1
    gkw = {'width_ratios': [1,1,1]} if len(datas)>2 else {'width_ratios': [1]}
    
    fig, axs = plt.subplots(num_provinces, num_columns, figsize=(10*num_columns, num_provinces*5), gridspec_kw=gkw)
    tick_indices = range(0, len(date_points), 10)
    
    for i, province in enumerate(province_names):
        if num_columns>1:
            ax_left, ax_middle, ax_right = axs[i]
        
            ax_left.scatter(date_points, obs_value[:,i,0], label=f"{province} - Ground Truth", color="black", s=8, marker='o')
            ax_left.scatter(date_points, pre_value[:,i,0], label=f"{province} - Prediction", color="red", s=8, marker='o')
            ax_left.set_xlabel("Date (days)", fontsize=12)
            ax_left.set_ylabel("Number of individuals", fontsize=15)
            ax_left.legend(fontsize=15)
            ax_left.set_xticks(tick_indices)
            ax_left.set_xticklabels([date_points[k] for k in tick_indices], rotation=0, ha='right')
            
            ax_middle.scatter(date_points, obs_value[:,i,0], label=f"{province} - Ground Truth", color="black", s=8, marker='o')
            ax_middle.scatter(date_points, phy_value[:,i,0], label=f"{province} - Causal Inference", color="green", s=8, marker='o')
            ax_middle.set_xlabel("Date (days)", fontsize=12)
            ax_middle.set_ylabel("Number of individuals", fontsize=15)
            ax_middle.legend(fontsize=15)
            ax_middle.set_xticks(tick_indices)
            ax_middle.set_xticklabels([date_points[k] for k in tick_indices], rotation=0, ha='right')
            
            ax_right.scatter(date_points, obs_value[:,i,0], label=f"{province} - Ground Truth", color="black", s=8, marker='o')
            ax_right.scatter(date_points, mix_value[:,i,0], label=f"{province} - Mix", color="blue", s=8, marker='o')
            ax_right.set_xlabel("Date (days)", fontsize=12)
            ax_right.set_ylabel("Number of individuals", fontsize=15)
            ax_right.legend(fontsize=15)
            ax_right.set_xticks(tick_indices)
            ax_right.set_xticklabels([date_points[k] for k in tick_indices], rotation=0, ha='right')
        
        else:
            ax_left = axs[i]
            
            ax_left.scatter(date_points, obs_value[:,i,0], label="Ground Truth", color="black", s=8, marker='o')
            ax_left.scatter(date_points, pre_value[:,i,0], label="Prediction", color="red", s=8, marker='o')
            ax_left.set_xlabel("Date (days)", fontsize=12)
            ax_left.set_ylabel("Number of individuals", fontsize=15)
            ax_left.set_title(province, fontsize=12)
            ax_left.legend(fontsize=15)
            ax_left.set_xticks(tick_indices)
            ax_left.set_xticklabels([date_points[k] for k in tick_indices], rotation=0, ha='center')
    
    plt.tight_layout()
    name = f'forecurve{num_columns}'
    plt.savefig(
        os.path.join(mkdirs(os.path.join(path, name)), f"{horizon}_{mode}_forecurve.pdf"),
        format="pdf",
        # dpi=600,
    )
    plt.close(fig)
    
    pass


def check_gradients(named_parameters, threshold_explosion=100, threshold_disappearing=1e-10):
    """
    Check the gradients of the model parameters to determine whether there is an exploding or vanishing gradient.

    Args:
    - named_parameters: model.named_parameters()
    - threshold_explosion (float): exploding gradient threshold, default: 100
    - threshold_disappearing (float): vanishing gradient threshold, default: 1e-5
    """
    for name, param in named_parameters:
        if param.grad is not None:
            grad_norm = param.grad.norm()

            # Check for exploding gradients
            if grad_norm > threshold_explosion:
                logger.warning(f"Gradient Explosion detected in layer {name} with norm {grad_norm:.30e}.")
            # Check for vanishing gradients
            elif grad_norm < threshold_disappearing:
                logger.warning(f"Gradient Disappearing detected in layer {name} with norm {grad_norm:.30e}.")
        else:
            logger.warning(f"Layer: {name} | No gradient computed.")


def plot_lrcurve(lrs, path):
    fig = plt.figure(figsize=(16, 9))

    plt.plot(lrs['epoch'], lrs['lr'], marker='o', linestyle='-', color='b', label=f"Learning Rate over Epochs - {len(lrs['lr'])}")

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    
    plt.legend()
    
    plt.savefig(os.path.join(path, "learning_rate.pdf"), format="pdf", dpi=300)
    plt.close(fig)
    
    pass


def plot_losscurve_separately(all_losses, path, early_stop=None):
    losses, losses_pre, losses_phy, losses_phy_y = all_losses

    train_loss, train_loss_pre, train_loss_phy, train_loss_phy_y = (
        losses["training"],
        losses_pre["training"],
        losses_phy["training"],
        losses_phy_y["training"],
    )
    validate_loss, validate_loss_pre, validate_loss_phy, validate_loss_phy_y = (
        losses["validation"],
        losses_pre["validation"],
        losses_phy["validation"],
        losses_phy_y["validation"],
    )

    loss_list = [
        train_loss,
        validate_loss,
        train_loss_pre,
        train_loss_phy,
        train_loss_phy_y,
        validate_loss_pre,
        validate_loss_phy,
        validate_loss_phy_y,
    ]
    losslabel_list = [
        "Training",
        "Validation",
        "Training pre",
        "Training phy",
        "Training phy_y",
        "Validation pre",
        "Validation phy",
        "Validation phy_y",
    ]
    
    df = pd.DataFrame(
        {
            "training": train_loss,
            "training_pre": train_loss_pre,
            "training_phy": train_loss_phy,
            "training_phy_y": train_loss_phy_y, 
            "validation": validate_loss,
            "validation_pre": validate_loss_pre,
            "validation_phy": validate_loss_phy,
            "validation_phy_y": validate_loss_phy_y,
        }
    )
        
    markdown_table = tabulate(df, headers='keys', tablefmt='github', colalign=["center"], showindex=True)
    with open(os.path.join(path, "losses.md"), "w") as f:
        f.write(markdown_table)

    epochs = np.arange(len(train_loss))

    plot_early_stop = early_stop is not None and early_stop > 0

    num_plots = len(loss_list)
    fig, axes = plt.subplots(num_plots, 2, figsize=(16, 4 * num_plots), constrained_layout=True)

    for i in range(len(loss_list)):
        for j in range(2):
            if plot_early_stop:
                if j == 0:
                    axes[i, j].plot(epochs[:-early_stop], loss_list[i][:-early_stop], label=f"{losslabel_list[i]}", color="black")
                    axes[i, j].plot(epochs[-early_stop:], loss_list[i][-early_stop:], label="Early_stop", color="green")
                    axes[i, j].set_ylabel("Loss")

                    ax_inset = inset_axes(axes[i, j], width="40%", height="30%", loc="upper center", borderpad=2)
                    ax_inset.plot(epochs[-early_stop:], loss_list[i][-early_stop:], label="Early_stop", color="red")
                    ax_inset.set_title("Early Stop Region")

                elif j == 1:
                    loss_array = np.array(loss_list[i])
                    safe_loss = np.where(loss_array > 0, loss_array, 1e-10)
                    axes[i, j].plot(
                        epochs[:-early_stop],
                        np.log(safe_loss[:-early_stop]),
                        label=f"{losslabel_list[i]} [log]",
                        color="orange",
                    )
                    axes[i, j].plot(
                        epochs[-early_stop:],
                        np.log(safe_loss[-early_stop:]),
                        label="Early_stop",
                        color="green",
                    )
                    axes[i, j].set_ylabel("Log Loss")

                    ax_inset = inset_axes(axes[i, j], width="30%", height="20%", loc="upper center", borderpad=2)
                    ax_inset.plot(epochs[-early_stop:], np.log(safe_loss[-early_stop:]), label="Early_stop", color="red")
                    ax_inset.set_title("Early Stop Region [Log]")
            else:
                if j == 0:
                    axes[i, j].plot(epochs, loss_list[i], label=f"{losslabel_list[i]}", color="black")
                    axes[i, j].set_ylabel("Loss")
                elif j == 1:
                    loss_array = np.array(loss_list[i])
                    safe_loss = np.where(loss_array > 0, loss_array, 1e-10)
                    axes[i, j].plot(
                        epochs,
                        np.log(safe_loss),
                        label=f"{losslabel_list[i]} [log]",
                        color="orange",
                    )
                    axes[i, j].set_ylabel("Log Loss")

            axes[i, j].set_xlabel("Epoch")
            axes[i, j].legend(loc="upper right")

    plt.savefig(os.path.join(path, "losses.pdf"), format="pdf", dpi=300)
    plt.close(fig)

    pass


def save_foredata(path, mode, names, files):
    dpath = os.path.join(path, "foredata")
    os.makedirs(dpath, exist_ok=True)
    for i,name in enumerate(names):
        np.save(os.path.join(dpath, f'{mode}_{name}'), files[i].cpu().detach().numpy())
        
    pass


def save_foreerror(path, errors, headers, filename):
    table_str = tabulate(errors, headers=headers, tablefmt='grid', colalign=["center"] * len(headers), floatfmt=".4f")

    errors_table_path = os.path.join(path, filename)
    with open(errors_table_path, 'w') as f:
        f.write(table_str)
        
    pass


def plot_foreepiparams(path, mode, epiparams, horizon, prov_names, time_ratio, offset, num_cols=7):
    date_points = DATE.DATE_LIST[(time_ratio[0]+time_ratio[1]+offset[0]):(time_ratio[0]+time_ratio[1]+time_ratio[2]-offset[1]+1)]
    epiparams = epiparams.cpu().detach().numpy()
    
    betas, gammas = epiparams[:, :, 0:1], epiparams[:, :, 1:2]
    if epiparams.shape[-1]>2:
        cs = epiparams[:, :, 2:]
        plot_R0(path, mode, betas, gammas, cs, prov_names, date_points, horizon)
    else:
        plot_R0(path, mode, betas, gammas, None, prov_names, date_points, horizon)
        return 

    _, nodes, _ = cs.shape  # cs (time_steps, nodes, nodes)
    num_rows = (len(date_points) + num_cols - 1) // num_cols
    node_points= np.arange(nodes)

    fig = plt.figure(figsize=(num_cols * 6, num_rows * 6)) 
    gs = GridSpec(num_rows, num_cols, figure=fig)

    for idx, date in enumerate(date_points):
        t = idx
        row = idx // num_cols
        col = idx % num_cols
        ax = plt.subplot(gs[row, col])

        im = ax.imshow(cs[t], cmap='viridis_r', interpolation='nearest')
        
        ax.set_xticks(node_points)
        ax.set_yticks(node_points)

        ax.set_xticklabels(node_points)
        ax.set_yticklabels(node_points)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        ax.set_title(date, fontsize=12)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(mkdirs(os.path.join(path, 'foreparams')), f"{horizon}_{mode}_foreepiparams.pdf"),
        format="pdf",
        # dpi=600,
    )
    plt.close(fig)
    
    pass


def plot_losscurve(tloss, path, estop):
    fig = plt.figure(figsize=(16, 9))

    plt.plot(np.arange(len(tloss)), tloss, label="Loss", color="black")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(os.path.join(path, "losses.pdf"), format="pdf", dpi=300)

    plt.close(fig)

    # save losses to csv file
    df = pd.DataFrame(
        {
            "loss": tloss
        }
    )
        
    markdown_table = tabulate(df, headers='keys', tablefmt='github', colalign=["center"], showindex=True)
    with open(os.path.join(path, "losses.md"), "w") as f:
        f.write(markdown_table)

    pass
