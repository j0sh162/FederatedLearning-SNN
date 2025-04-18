# import flwr as fl
# import torch
# from typing import Dict
# from collections import OrderedDict
# from flwr.common import NDArrays,Scalar
# from FL.CNN import Net,train,test
# #our models will send us NUMPY Array parameters

# class FlowerClient(fl.client.NumPyClient):
#     def __init__(self,
#                  trainloader,
#                  valloader,
#                  num_classes) -> None:
#         super().__init__()
#         self.trainloader = trainloader
#         self.valloader = valloader
#         self.model = Net(num_classes)
#         #run on GPU if available else CPU
#         #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.device = torch.device("cpu")
#         #print("client innit please :(")
#         #params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
#         #print("parameters: ",params)
#
# def set_parameters(self,parameters):
#     print("set parameters please :(")
#     parameters_dict = zip(self.model.state_dict().keys(),parameters)
#     state_dict = OrderedDict({k: torch.tensor(v) for k,v in parameters_dict}) #convert numpy array into tensor representation
#     self.model.load_state_dict(state_dict, strict=True)
#
# def get_parameters(self,config: Dict[str, Scalar]):
#     print("get parameters please :(")
#     return [val.cpu().numpy() for _,val in self.model.state_dict().items()]
#         def get_parameters(self, config: Dict[str, Scalar]):
#             params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
#             return params
#
# def fit(self,parameters,config):
#     """Update the weights of the model"""
#     #copy parameters sent by server into client's local model
#     print("fit function ")
#     print("in client parameters: ",parameters)
#     self.set_parameters(parameters)
#
#     #training model locally
#     lr=config.client.lr
#     momentum=config.client.momentum
#     epochs = config.client.local_epochs
#     optim=torch.optim.SGD(self.model.parameters(),lr=lr,momentum=momentum)
#
#     train(self.model,self.trainLoader,optim,epochs,self.device)
#     #return updated model
#     return self.get_parameters({}),len(self.trainLoader),{}
#
# def evaluate(self,parameters: NDArrays, config: Dict[str, Scalar]):
#     self.set_parameters(parameters)
#     loss,accuracy=test(self.model,self.valloader,self.device)
#     return float(loss),len(self.valloader),{'accuracy':accuracy}
#
# def generate_client_fn(trainloaders,valloaders,num_classes):
#     """spawning clients for simulation"""
#     def client_fn(clientID: str):
#         print("client function with id ",clientID)
#         return FlowerClient(trainloader=trainloaders[int(clientID)],
#                             valloader=valloaders[int(clientID)]
#                             ,num_classes=num_classes)
#     return client_fn
from collections import OrderedDict
from typing import Dict

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from hydra.utils import instantiate

from FL.CNN import test, train


# from model import Net, train, test
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, model_cfg) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = instantiate(model_cfg)
        # run on GPU if available else CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        # print("client innit please :(")
        # params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        # print("parameters: ",params)

    # Copies the parameters of the global model into your local
    def set_params(self, parameters):
        print("set parameters please :(")
        parameters_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in parameters_dict}
        )  # convert numpy array into tensor representation
        self.model.load_state_dict(state_dict, strict=True)

    # Gets the parameters of yout local model
    def get_parameters(self, config: Dict[str, Scalar]):
        print("get parameters please :(")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # Receives from server parameters from global model and set of instructions and trains the model (standard pytorch training method)
    def fit(self, parameters, config):
        """Update the weights of the model - returns parameters of locally trained model"""
        # copy parameters sent by server into client's local model
        print("fit function ")
        # print("in client parameters: ", parameters)
        self.set_params(parameters)

        # training model locally
        print(
            "lr ",
            config["lr"],
            " momentum ",
            config["momentum"],
            " local_epochs ",
            config["local_epochs"],
        )
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        train(self.model, self.trainloader, optim, epochs, self.device)
        # return updated model
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Receive parameters from global model and evaluate local validation set and return wanted information in this case: loss/accuracy"""
        self.set_params(parameters)
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
        ).to_client()

    return client_fn
