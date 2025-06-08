from typing import Dict
import flwr as fl
from flwr.common import NDArrays,Scalar
import torch
from collections import OrderedDict
from FL.CNN import train,test
from hydra.utils import instantiate
# from model import Net, train, test
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, model_cfg) -> None:
        super().__init__()
        self.trainloader = trainloader  #train loader assigned to this specific client
        self.valloader = valloader #validation loader assigned to this specific client
        self.model = instantiate(model_cfg) #model used by this specific client

        # run on GPU if available else CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Sets parameters we get from server
    def set_params(self, parameters):
        parameters_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in parameters_dict})  # convert numpy array into tensor representation
        self.model.load_state_dict(state_dict, strict=True)

    #Gets the parameters of local model (from client's model)
    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]
    #Receives from server parameters from global model and set of instructions and trains the model (standard pytorch training method)
    def fit(self, parameters, config):
        """Update the weights of the model - returns parameters of locally trained model"""
        # copy parameters sent by server into client's local model
        print("fit function ")
        self.set_params(parameters)

        # training model locally
        print("lr ", config["lr"],  " momentum ", config["momentum"], " local_epochs ", config["local_epochs"])
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        train(self.model, self.trainloader, optim, epochs, self.device) #The model's train function (See CNN.py for reference)
        # return updated model
        return self.get_parameters({}), len(self.trainloader), {} # in {you can send information about accuracy/energy used by individual clients}
                                                                  # So here we would include information about energy consumption of individual client

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Receive parameters from global model and evaluate local validation set and return wanted information in this case: loss/accuracy"""
        #In short evaluate how global model performs on validatoin set of this particular client
        self.set_params(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader), {'accuracy': accuracy}


# Returns a function that spawns a client in the main
def generate_client_fn(trainloaders,valloaders,model_cfg):
    """spawning clients for simulation"""
    def client_fn(clientID: str):
        print("client function with id ",clientID)
        return FlowerClient(
                            trainloader=trainloaders[int(clientID)],
                            valloader=valloaders[int(clientID)],
                            model_cfg=model_cfg)
    return client_fn

