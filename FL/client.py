from collections import OrderedDict
from typing import Dict

import flwr as fl
import torch
from flwr.common import Context, NDArrays, Scalar
from hydra.utils import instantiate
from rich import print

from FL.training_utils import test, train
from SNN_Models import SNN_utils


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, model_cfg) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = instantiate(model_cfg)
        self.model_cfg = model_cfg
        # run on GPU if available else CPU
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
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
        if self.model_cfg._target_ == "SNN_Models.SNN.Net":
            SNN_utils.train(
                self.model, self.trainloader, optim, config["local_epochs"], self.device
            )
        elif self.model_cfg._target_ == "FL.CNN.Net":
            train(
                self.model, self.trainloader, optim, config["local_epochs"], self.device
            )
        # return updated model
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Receive parameters from global model and evaluate local validation set and return wanted information in this case: loss/accuracy"""
        self.set_params(parameters)
        if self.model_cfg._target_ == "SNN_Models.SNN.Net":
            loss, accuracy = SNN_utils.test(self.model, self.valloader, self.device)
        elif self.model_cfg._target_ == "FL.CNN.Net":
            loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": accuracy}


# Returns a function that spawns a client in the main
def generate_client_fn(trainloaders, valloaders, model_cfg):
    """spawning clients for simulation"""

    def client_fn(clientID: str):
        # TODO Add context: Context as argument as this way is deprecated
        print("client function with id ", clientID)
        return FlowerClient(
            trainloader=trainloaders[int(clientID)],
            valloader=valloaders[int(clientID)],
            model_cfg=model_cfg,
        ).to_client()

    return client_fn
