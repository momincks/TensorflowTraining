import importlib

def ParseMultiLoss(loss_name):

    if "+" not in loss_name:
        raise ValueError("please split loss names by using '+'")

    return loss_name.lower().split("+")

def ImportLoss(loss_name):
    return importlib.import_module(f".func.{loss_name}")