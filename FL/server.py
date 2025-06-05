"""Local server that trains the model among all clients"""

from collections import OrderedDict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ray import client
from biograd.online_error_functions import cross_entropy_loss_error_function

from tqdm import tqdm
import time

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


def get_evaluate_fn(model_cfg, testLoader):
    def evaluate_fn(server_round: int, parameters, config):
        local_model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        local_model_cfg["error_func"] = cross_entropy_loss_error_function
        model = instantiate(local_model_cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        # dictionary = model.state_dict()
        print("client", server_round)
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
        # TODO Adjust for use with different models
        loss, accuracy = test(model, testLoader, device)
        return loss, {"accuracy": accuracy}

    return evaluate_fn

def test(net, testloader, device: str):
    with torch.no_grad():
        test_num = len(testloader.dataset)
        test_correct = 0
        test_start = time.time()
        for data in tqdm(testloader, desc="Test", leave=False):
            event_img, labels = data
            event_img, labels = event_img.to(device), labels.to(device)
            predict_label = net.test(event_img)
            test_correct += ((predict_label == labels).sum().to("cpu")).item()
        test_end = time.time()
        test_accuracy = test_correct / test_num
        print("Test Accuracy %.4f Test Time: %.1f" % (test_accuracy, test_end - test_start), end=" ")
        return 0.5, test_accuracy