from collections import OrderedDict
import flwr as fl
import torch
from torch import nn,optim
from utils import AverageMeter, accuracy
import numpy as np
from SNN_Models.snn_spide_mnist_lenet import SNNSPIDEConvMultiLayerNet



def get_model_config(args_dict):
    config = {}
    config["MODEL"] = {}
    config["MODEL"]["c_in"] = 1  # MNIST specific
    config["MODEL"]["c_hidden"] = 15
    config["MODEL"]["c_s1"] = 40
    config["MODEL"]["h_hidden"] = 28 # This might need adjustment based on SNN layers
    config["MODEL"]["w_hidden"] = 28 # This might need adjustment
    config["MODEL"]["fc_num"] = 300
    config["MODEL"]["num_classes"] = 10  # MNIST specific
    config["MODEL"]["kernel_size_x"] = 5
    config["MODEL"]["stride_x"] = 1
    config["MODEL"]["padding_x"] = 2
    config["MODEL"]["kernel_size_s"] = 5
    config["MODEL"]["time_step"] = args_dict["time_step"]
    config["MODEL"]["time_step_back"] = args_dict["time_step_back"]
    config["MODEL"]["vth"] = args_dict["vth"]
    if args_dict["vth_back"] != 0:
        config["MODEL"]["vth_back"] = args_dict["vth_back"]
    config["MODEL"]["dropout"] = args_dict["drop"]
    config["MODEL"]["leaky"] = None # Or get from args_dict if you add it
    return config

# def train_fl(
#     model,
#     trainloader,
#     device,
#     epochs,
#     args_dict,
# ):
#     model.train()
#     criterion = nn.CrossEntropyLoss()
#     if args_dict["optimizer"] == "SGD":
#         optimizer = optim.SGD(
#             model.parameters(),
#             lr=args_dict["lr"] / args_dict["scale_factor"],
#             momentum=args_dict["momentum"],
#             weight_decay=args_dict["weight_decay"],
#         )
#     else:
#         optimizer = optim.Adam(
#             model.parameters(),
#             lr=args_dict["lr"] / args_dict["scale_factor"],
#             weight_decay=args_dict["weight_decay"],
#             betas=(0.9, 0.999),
#         )

#     epoch_losses = []
#     epoch_accs = []

#     for epoch in range(epochs):
#         losses = AverageMeter()
#         top1 = AverageMeter()
#         for inputs, targets in trainloader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs) # Assuming model handles time_steps internally
#             loss = criterion(outputs, targets)

#             prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
#             losses.update(loss.item(), inputs.size(0))
#             top1.update(prec1.item(), inputs.size(0))

#             scaled_loss = loss * args_dict["scale_factor"]
#             optimizer.zero_grad()
#             scaled_loss.backward()
#             optimizer.step()
#         print(f"  Client Epoch {epoch+1}/{epochs} -> Loss: {losses.avg:.4f}, Acc: {top1.avg:.2f}%")
#         epoch_losses.append(losses.avg)
#         epoch_accs.append(top1.avg)
#     return np.mean(epoch_losses), np.mean(epoch_accs)

def train_fl(
    model,
    trainloader,
    device,
    epochs,
    args_dict,
):
    model.train()
    criterion = nn.CrossEntropyLoss()
    if args_dict["optimizer"].lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args_dict["lr"] / args_dict["scale_factor"],
            momentum=args_dict["momentum"],
            weight_decay=args_dict["weight_decay"],
        )
    elif args_dict["optimizer"].lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args_dict["lr"] / args_dict["scale_factor"],
            weight_decay=args_dict["weight_decay"],
            # betas=(0.9, 0.999), # Default Adam betas
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args_dict['optimizer']}")

    epoch_losses = []
    epoch_accs = []

    for epoch in range(epochs):
        losses = AverageMeter()
        top1 = AverageMeter()
        for batch_idx, (inputs, targets) in enumerate(iter(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)

            # NMNIST with ToFrame and PadTensors(batch_first=False) gives (T, B, C, H, W)
            # If your SNN expects (B, C, H, W) and processes time internally,
            # you might need to reshape or select frames.
            # Example: if model processes one frame at a time and averages outputs:
            # B, T, C, H, W = inputs.shape # if batch_first=True
            # T, B, C, H, W = inputs.shape # if batch_first=False (current case)
            # For simplicity, let's assume the model's forward can handle (T,B,C,H,W)
            # or you adapt it. If it expects (B,C,H,W), you might do:
            # inputs = inputs.mean(dim=0) # Average over time, or select first frame inputs[0]
            # This part is CRITICAL and depends on your SNNSPIDEConvMultiLayerNet's forward pass.

            outputs = model(inputs) # Model handles its internal time_steps
            loss = criterion(outputs, targets)

            prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5)) # Ensure accuracy fn is compatible
            losses.update(loss.item(), inputs.size(1) if inputs.ndim > 1 else inputs.size(0)) # inputs.size(1) is batch size if T,B,C,H,W
            top1.update(prec1.item(), inputs.size(1) if inputs.ndim > 1 else inputs.size(0))

            scaled_loss = loss * args_dict["scale_factor"]
            optimizer.zero_grad()
            scaled_loss.backward()
            optimizer.step()
        print(f"  Client Epoch {epoch+1}/{epochs} -> Loss: {losses.avg:.4f}, Acc: {top1.avg:.2f}%")
        epoch_losses.append(losses.avg)
        epoch_accs.append(top1.avg)
    return np.mean(epoch_losses), np.mean(epoch_accs)


# Modified test function for Flower client and server-side evaluation
# def test_fl(model, testloader, device, args_dict):
#     model.eval()
#     criterion = nn.CrossEntropyLoss()
#     losses = AverageMeter()
#     top1 = AverageMeter()

#     with torch.no_grad():
#         for inputs, targets in testloader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs) # Assuming model handles time_steps internally
#             loss = criterion(outputs, targets)

#             prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
#             losses.update(loss.item(), inputs.size(0))
#             top1.update(prec1.item(), inputs.size(0))
#     return losses.avg, top1.avg

def test_fl(model, testloader, device, args_dict): # args_dict might not be needed if model config is fixed
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Similar consideration for input shape as in train_fl
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(1) if inputs.ndim > 1 else inputs.size(0))
            top1.update(prec1.item(), inputs.size(1) if inputs.ndim > 1 else inputs.size(0))
    return losses.avg, top1.avg


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, args_dict):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = 'cpu'
        self.args_dict = args_dict

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # print(f"Client fitting on {len(self.trainloader.dataset)} samples...")
        # Pass relevant items from server `config` if needed, e.g., current_round
        loss, acc = train_fl(
            self.model,
            self.trainloader,
            self.device,
            epochs=self.args_dict["epochs_per_round"],
            args_dict=self.args_dict,
        )
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"accuracy": acc, "loss": loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        print(f"Client evaluating on {len(self.testloader.dataset)} samples...")
        loss, accuracy_val = test_fl(
            self.model, self.testloader, self.device, self.args_dict
        )
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy_val)}
    

def client_fn(cid: str, client_dataloaders, args_dict) -> fl.client.Client: # Changed return type hint
    """Create a Flower client instance for simulation."""
    model_cfg = get_model_config(args_dict)
    model = SNNSPIDEConvMultiLayerNet(model_cfg)
    trainloader, testloader = client_dataloaders
    numpy_client = FlowerClient(model, trainloader[int(cid)], testloader[int(cid)], args_dict)
    return numpy_client.to_client()


def generate_client_fn(train_loader,test_loader,cfg):
    #TODO make cfg become the arg_dict
    args_dict = {
    "dataset": "mnist",
    "path": "./data",
    "workers": 2,
    "time_step": 30,
    "time_step_back": 100,
    "vth": 1.0,
    "vth_back": 0.5,
    "epochs_per_round": 1, # Local epochs per FL round
    "train_batch": 128,
    "test_batch": 200,
    "lr": 0.1,
    "drop": 0.2,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "optimizer": "SGD",
    "scale_factor": 100.0,
    "manualSeed": None,
    "gpu_id": "0",
    "num_clients": 5,
    "num_rounds": 10,
    }
    client_dataloaders = (train_loader,test_loader)

    return lambda cid: client_fn(cid, client_dataloaders, args_dict)



def get_evaluate_fn(model_cfg,server_testloader):
    args_dict = {
    "dataset": "mnist",
    "path": "./data",
    "workers": 2,
    "time_step": 30,
    "time_step_back": 100,
    "vth": 1.0,
    "vth_back": 0.5,
    "epochs_per_round": 1, # Local epochs per FL round
    "train_batch": 128,
    "test_batch": 200,
    "lr": 0.1,
    "drop": 0.2,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "optimizer": "SGD",
    "scale_factor": 100.0,
    "manualSeed": None,
    "gpu_id": "0",
    "num_clients": 5,
    "num_rounds": 10,
    }
    
    device = 'cpu'
    def evaluate_fn(server_round: int, parameters: fl.common.NDArrays, config):
        model_cfg = get_model_config(args_dict)
        model = SNNSPIDEConvMultiLayerNet(model_cfg)
        # model = torch.nn.DataParallel(model) # If using DataParallel
        
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

        loss, acc = test_fl(model, server_testloader, device, args_dict)
        print(f"Server-side evaluation round {server_round}: Loss {loss:.4f}, Accuracy {acc:.2f}%")
        return loss, {"accuracy": acc}
    
    return evaluate_fn