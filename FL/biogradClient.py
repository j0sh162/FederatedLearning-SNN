from collections import OrderedDict
from typing import Dict

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from hydra.utils import instantiate
from rich.logging import RichHandler

from tqdm import tqdm, trange
import time

from torch import nn
from biograd.online_error_functions import cross_entropy_loss_error_function


dt = 5
unlimited_mem = False

# Define SNN parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_shape = 2 * 34 * 34
out_dim = 10
snn_param = {'hidden_layer': [(0.3, 0.3, 0.3, 1.0),
                              (0.3, 0.3, 0.3, 1.0)],
             'out_layer': (0.3, 0.3, 0.3, 1.0)}
soft_error_step = 19
sleep_spike_ts = 50

# Define Training parameters
val_size = 10000
train_batch_size = 128
sleep_batch_size = 128
test_batch_size = 1
epoch = 100
save_epoch = 1
lr = 1.0e-3
sleep_oja_power = 2.0
sleep_lr = 1.0e-4 / 3.

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, model_cfg) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        model_cfg = model_cfg.copy()
        self.model = instantiate(model_cfg)
        self.model.error_func = cross_entropy_loss_error_function
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
        train(self.model, self.trainloader, optim, config["local_epochs"], self.device)
        # return updated model
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Receive parameters from global model and evaluate local validation set and return wanted information in this case: loss/accuracy"""
        self.set_params(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return loss, len(self.valloader), {"accuracy": accuracy}


# Returns a function that spawns a client in the main
def generate_client_fn(trainloaders, valloaders, model_cfg):
    """spawning clients for simulation"""

    def client_fn(clientID: str):
        # TODO Add context: Context as argument as this way is deprecated
        # print(f"Generate client function with id {clientID}")
        return FlowerClient(
            trainloader=trainloaders[int(clientID)],
            valloader=valloaders[int(clientID)],
            model_cfg=model_cfg,
        )

    return client_fn

def train(net, train_loader, optimizer, epochs, device: str):
    feedback_angle, feedback_ratio = [], []
    angle_list, ratio_list = net.compute_feedback_angle_ratio()
    feedback_angle.append(angle_list)
    feedback_ratio.append(ratio_list)
    # Start training
    with torch.no_grad():
        for ee in trange(epoch, desc="Epoch"):
            # Training
            train_num = len(train_loader.dataset)
            train_correct = 0
            train_start = time.time()
            for data in tqdm(train_loader, desc="Train", leave=False):
                event_img, labels = data
                labels_one_hot = nn.functional.one_hot(labels, num_classes=10).float()
                event_img, labels, labels_one_hot = event_img.to(device), labels.to(device), labels_one_hot.to(device)
                predict_label, hid_fwd_states, hid_fb_states, out_fwb_state, out_fb_state, fb_step = net.train_online(
                    event_img, labels_one_hot, soft_error_step)
                net.train_update_parameter(hid_fwd_states, hid_fb_states, out_fwb_state, out_fb_state, fb_step, lr)
                train_correct += ((predict_label == labels).sum().to("cpu")).item()

                # Put network to sleep for feedback training
                net.sleep_feedback_update(sleep_batch_size, sleep_spike_ts, sleep_oja_power, sleep_lr)

                # Compute angle and ratio between feedback weight and forward weight after each update
                angle_list, ratio_list = net.compute_feedback_angle_ratio()
                feedback_angle.append(angle_list)
                feedback_ratio.append(ratio_list)

            train_end = time.time()
            train_accuracy = train_correct / train_num
            print("Epoch %d Training Accuracy %.4f Train time: %.1f" % (ee, train_accuracy, train_end - train_start), end=" ")

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

def model_to_parameters(model):
    from flwr.common.parameter import ndarrays_to_parameters

    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)
    print("model extracted")
    return parameters