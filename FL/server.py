"""Local server that trains the model among all clients"""
from omegaconf import DictConfig
from ray import client

from FL.CNN import Net,test
from collections import OrderedDict
import torch
def get_on_fit_config(config: DictConfig):
    def fit_config_function(server_round: int):
        #if server_round == 50: #Example on how we can manimulate learning rate in later stage
        #    lr = config.lr/10
        return {'lr': config.client.lr,'momentum':config.client.momentum,'local_epochs':config.client.local_epochs}
    return fit_config_function

def get_evaluate_fn(num_classes: int,testLoader):
    def evaluate_fn(server_round: int, parameters, config):
        model = Net(num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = torch.device("cpu")
        #dictionary = model.state_dict()
        print("client", server_round)
        #print("length before: ", len(dictionary))
        #for k,v in dictionary.items():
        #    print(f"{k}: {v}")

        parameters_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k:torch.Tensor(v) for k,v in parameters_dict})

        #print("length: ",len(state_dict))
        #for k,v in state_dict.items():
            #print("k: ", k," v: ",v)
        model.load_state_dict(state_dict,strict=True)
        #print("parameters in server: ", parameters, "client", server_round)
        loss, accuracy = test(model,testLoader,device)
        return loss, {"accuracy":accuracy}
    return evaluate_fn