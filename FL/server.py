"""Local server that trains the model among all clients"""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import torch
from flwr.common import Metrics
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from ray import client

from FL.CNN import Net
from FL.training_utils import test
from SNN_Models import EventProp, SNN_utils, Spide

# from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("flwr")
# global_server_round = 0
# # writer cannot be initialized here, because the seed is not set yet


# def get_fit_metrics_aggregation_fn():  # -> Callable[..., dict[str, float]]:
#     seed = torch.initial_seed()
#     writer = SummaryWriter(log_dir=f"runs/federated_server/{seed}")

#     def fit_metrics_aggregation_fn(metrics):
#         # Extract metrics and num_examples from results
#         total_examples = sum(num_examples for num_examples, _ in metrics)
#         avg_loss = (
#             sum(num_examples * m.get("loss", 0) for num_examples, m in metrics)
#             / total_examples
#         )
#         avg_acc = (
#             sum(num_examples * m.get("accuracy", 0) for num_examples, m in metrics)
#             / total_examples
#         )

#         writer.add_scalar("client/train/loss", avg_loss, global_server_round)
#         writer.add_scalar("client/train/accuracy", avg_acc, global_server_round)
#         print("Cleint Fit metrics agg:", {"loss": avg_loss, "accuracy": avg_acc})
#         return {"loss": avg_loss, "accuracy": avg_acc}

#     return fit_metrics_aggregation_fn


# def get_evaluate_metrics_aggregation_fn():
#     seed = torch.initial_seed()
#     writer = SummaryWriter(log_dir=f"runs/federated_server/{seed}")

#     def evaluate_metrics_aggregation_fn(metrics):
#         total_examples = sum(num_examples for num_examples, _ in metrics)
#         avg_loss = (
#             sum(num_examples * m.get("loss", 0) for num_examples, m in metrics)
#             / total_examples
#         )
#         avg_acc = (
#             sum(num_examples * m.get("accuracy", 0) for num_examples, m in metrics)
#             / total_examples
#         )

#         writer.add_scalar("client/valid/loss", avg_loss, global_server_round)
#         writer.add_scalar("client/valid/accuracy", avg_acc, global_server_round)
#         print("Client Evaluate metrics agg:", {"loss": avg_loss, "accuracy": avg_acc})
#         return {"loss": avg_loss, "accuracy": avg_acc}

#     return evaluate_metrics_aggregation_fn


def get_on_fit_config(config: DictConfig):
    def fit_config_function(server_round: int):
        global global_server_round
        global_server_round = server_round
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
def get_evaluate_fn(cfg, testLoader):
    # seed = torch.initial_seed()
    # writer = SummaryWriter(log_dir=f"runs/federated_server/{seed}")

    def evaluate_fn(server_round: int, parameters, config):
        model = instantiate(cfg.model)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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

        # Save model every evaluation round
        save_path = HydraConfig.get().runtime.output_dir
        model_path = Path(save_path) / "state_dict.pth"
        torch.save(model.state_dict(), model_path)
        # print("parameters in server: ", parameters, "client", server_round)

        if cfg.model._target_ == "SNN_Models.SNN.Net":
            loss, accuracy = SNN_utils.test(model, testLoader, device)
        elif cfg.model._target_ == "SNN_Models.EventProp.SNN":
            loss, accuracy = EventProp.test(model, testLoader, device)
        elif cfg.model._target_ == "SNN_Models.Spide.SNNSPIDEConvMultiLayerNet":
            loss, accuracy = Spide.test_fl(model, testLoader, device)
        elif cfg.model._target_ == "FL.CNN.Net":
            loss, accuracy = test(model, testLoader, device)
        else:
            raise ValueError(f"Unsupported model configuration: {cfg.model}")

        # writer.add_scalar("server/test/loss", loss, server_round)
        # writer.add_scalar("server/test/accuracy", accuracy, server_round)

        return loss, {"accuracy": accuracy}

    return evaluate_fn
