import random
from collections import OrderedDict
from typing import Dict

import flwr as fl
import numpy as np
import torch
from flwr.common import Context, NDArrays, Scalar
from hydra.utils import instantiate
from rich.logging import RichHandler

from FL.training_utils import test, train
from SNN_Models import EventProp, SNN_utils, Spide


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, model_cfg, seed) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = instantiate(model_cfg)
        self.model_cfg = model_cfg
        self.seed = seed
        # Seeds have to be set anew because the client is spawned in a new process
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # run on GPU if available else CPU
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    # Copies the parameters of the global model into your local
    def set_params(self, parameters):
        parameters_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in parameters_dict}
        )  # convert numpy array into tensor representation
        self.model.load_state_dict(state_dict, strict=True)

    # Gets the parameters of your local model
    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # Receives from server parameters from global model and set of instructions
    # and trains the model (standard pytorch training method)
    def fit(self, parameters, config):
        """Update the weights of the model - returns parameters of locally trained model"""
        # copy parameters sent by server into client's local model
        self.set_params(parameters)

        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["lr"],
            betas=[0.9, 0.999],
            # betas=config["betas"],
        )
        # if config["model"]["name"] == "SNN":
        if self.model_cfg._target_ == "SNN_Models.SNN.Net":
            SNN_utils.train(
                self.model, self.trainloader, optim, config["local_epochs"], self.device
            )
        elif self.model_cfg._target_ == "SNN_Models.EventProp.SNN":
            EventProp.train(
                self.model, optim, self.trainloader, config["local_epochs"], self.device
            )
        elif self.model_cfg._target_ == "SNN_Models.Spide.SNNSPIDEConvMultiLayerNet":
            Spide.train_fl(
                self.model, self.trainloader, self.device, config["local_epochs"], optim
            )
        # elif self.model_cfg._target_ == "FL.CNN.Net":
        #     train(
        #         self.model, self.trainloader, optim, config["local_epochs"], self.device
        #     )
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Receive parameters from global model and evaluate local validation set and return wanted information in this case: loss/accuracy"""
        self.set_params(parameters)
        if self.model_cfg._target_ == "SNN_Models.SNN.Net":
            loss, accuracy = SNN_utils.test(self.model, self.valloader, self.device)
        elif self.model_cfg._target_ == "SNN_Models.EventProp.SNN":
            loss, accuracy = EventProp.test(self.model, self.valloader, self.device)
        elif self.model_cfg._target_ == "SNN_Models.Spide.SNNSPIDEConvMultiLayerNet":
            loss, accuracy = Spide.test_fl(self.model, self.valloader, self.device)
        elif self.model_cfg._target_ == "FL.CNN.Net":
            loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": accuracy}


# Returns a function that spawns a client in the main
def generate_client_fn(trainloaders, valloaders, model_cfg, seed):
    """spawning clients for simulation"""

    def client_fn(clientID: str):
        # TODO Add context: Context as argument as this way is deprecated
        # print(f"Generate client function with id {clientID}")
        return FlowerClient(
            trainloader=trainloaders[int(clientID)],
            valloader=valloaders[int(clientID)],
            model_cfg=model_cfg,
            seed=seed,
        ).to_client()

    return client_fn
