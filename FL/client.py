from collections import OrderedDict
from typing import Dict

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
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
        if not isinstance(self.model, torch.nn.Module):
            self.model = self.model()
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

        # DEBUG: Check model type and parameters
        print("DEBUG: Model type:", type(self.model))
        params = list(self.model.parameters())
        print("DEBUG: Number of model parameters:", len(params))
        for name, param in self.model.named_parameters():
            print(f"Param: {name}, requires_grad={param.requires_grad}, shape={param.shape}")

        if len(params) == 0:
            raise ValueError("The model has no trainable parameters! Check if it's correctly built.")

        # Initialize optimizer
        optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config["lr"],
            betas=[0.9, 0.999],
        )

        # Train the model
        SNN_utils.train(
            self.model, self.trainloader, optim, config["local_epochs"], self.device
        )

        # Return updated model parameters
        return self.get_parameters({}), len(self.trainloader), {}


    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Receive parameters from global model and evaluate local validation set and return wanted information in this case: loss/accuracy"""
        self.set_params(parameters)
        # if config["model"] == "SNN":
        if True:
            loss, accuracy = SNN_utils.test(self.model, self.valloader, self.device)
        else:
            loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": accuracy}


# Returns a function that spawns a client in the main
def generate_client_fn(trainloaders, valloaders, model_cfg):
    """spawning clients for simulation"""

    def client_fn(clientID: str):
        print("client function with id ", clientID)
        return FlowerClient(
            trainloader=trainloaders[int(clientID)],
            valloader=valloaders[int(clientID)],
            model_cfg=model_cfg,
        )

    return client_fn
