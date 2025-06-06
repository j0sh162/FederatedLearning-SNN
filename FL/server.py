"""Local server that trains the model among all clients"""

import logging
from collections import OrderedDict
from typing import List, Tuple

import torch
from flwr.common import Metrics
from hydra.utils import instantiate
from omegaconf import DictConfig
from ray import client

from FL.CNN import Net
from FL.training_utils import test
from SNN_Models import SNN_utils
from SNN_Models import EventProp,Spide

logger = logging.getLogger("rich")


def get_on_fit_config(config: DictConfig):
    def fit_config_function(server_round: int):
        # if server_round == 50: #Example on how we can manimulate learning rate in later stage
        #    lr = config.lr/10
        # return {'lr': config.client.lr,'momentum':config.client.momentum,'local_epochs':config.client.local_epochs}
        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
        }

    return fit_config_function


# TODO Make this general so then it nice easy and switch
def get_evaluate_fn(model_cfg, testLoader):
    def evaluate_fn(server_round: int, parameters, config):
        model = instantiate(model_cfg)
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        # device = torch.device("cpu")
        # dictionary = model.state_dict()
        # print("length before: ", len(dictionary))
        # for k,v in dictionary.items():
        #    print(f"{k}: {v}")

        parameters_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.as_tensor(v) for k, v in parameters_dict})
        # TODO Tensors with 0 are being turned into empty tensors (like [])

        # print("length: ",len(state_dict))
        # for k,v in state_dict.items():
        # print("k: ", k," v: ",v)

        model.load_state_dict(state_dict, strict=True)
        # print("parameters in server: ", parameters, "client", server_round)

        if model_cfg._target_ == "SNN_Models.SNN.Net":
            loss, accuracy = SNN_utils.test(model, testLoader, device)
        elif model_cfg._target_ == "SNN_Models.EventProp.SNN":
            loss,accuracy = EventProp.test(model,testLoader,device)
        elif model_cfg._target_ == "SNN_Models.Spide.SNNSPIDEConvMultiLayerNet":
            loss,accuracy = Spide.test_fl(model,testLoader,device)
        elif model_cfg._target_ == "FL.CNN.Net":
            loss, accuracy = test(model, testLoader, device)
        else:
            raise ValueError(f"Unsupported model configuration: {model_cfg}")
        return loss, {"accuracy": accuracy}

    return evaluate_fn
