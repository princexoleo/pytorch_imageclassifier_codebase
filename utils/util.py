"""
Created by Mazharul Islam Leon

"""
import torch
import numpy as np 
import os 
import torch 
import torch.nn as nn
from markdown import markdown
from .args import args


def test():
    """
        For testing purpose
    """
    print("[Utils]: Testing method call")

#-----------fixed the randomnase ------------- #
def fix_random_seed(seed=0):
    """To fix the random characteristic of torch & numpy """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.backend.cudnn.benchmark = False
            torch.backend.cudnn.deterministic = False
    else:
        if torch.cuda.is_available():
            torch.backend.cudnn.benchmark = True
            torch.backend.cudnn.deterministic = True

# ----------- Saving the mode --------------#
def save_mode(model, pth="../models/trained_models/"):
    """ Saves a torch model in two ways: to be retrained or validation only """
    pth = os.path.join(pth, args.dataset)
    if not os.path.exists(pth):
        os.makedirs(pth)
    # models for only used in inference
    m = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(m.state_dict(), os.path.join(pth, args.model+".pt"))
    print("\nModel Saved!")
    return

def data_loader(dataset_name,):
    """
    This is dataloader method
    params:
        dataset_name: it will take a dataset name 
    ret:
        return the trainloader, valloader and testloader
    """