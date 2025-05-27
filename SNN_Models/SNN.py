import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
import copy
import math
from SNN_Models.network_cells import OnlineHiddenCell, OnlineOutputCell
from SNN_Models.online_error_functions import cross_entropy_loss_error_function

class Net(nn.Module):
    """ Online Learning Network with Sleep Weight Mirror Feedback Learning """

    def __init__(self, input_dim=2 * 34 * 34, output_dim=10, hidden_dim_list=[500, 100], param_dict={'hidden_layer': [(0.6, 0.3, 0.3, 1.0),
                              (0.6, 0.3, 0.3, 1.0)],
             'out_layer': (0.6, 0.3, 0.3, 1.0)}, error_func=cross_entropy_loss_error_function, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """

        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dim_list (list): list of hidden layer dimension
            param_dict (dict): neuron parameter dictionary
            error_func (function): error function
            device (device): device
        """
        super(Net, self).__init__()

        self.hidden_cells = nn.ModuleList()
        # Init Hidden Layers
        for idx, hh in enumerate(hidden_dim_list, 0):
            forward_output_dim = hh
            if idx == 0:
                forward_input_dim = input_dim
            else:
                forward_input_dim = hidden_dim_list[idx - 1]

            self.hidden_cells.append(
                OnlineHiddenCell(nn.Linear(forward_input_dim, forward_output_dim).to(device),
                                 nn.Linear(output_dim, hh, bias=False).to(device),
                                 param_dict['hidden_layer'][idx],
                                 forward_input_dim, forward_output_dim))
        # Init Output Layer
        self.output_cell = OnlineOutputCell(nn.Linear(hidden_dim_list[-1], output_dim).to(device),
                                            param_dict['out_layer'],
                                            hidden_dim_list[-1], output_dim)

        # Init Feedback Connections
        feedback_weight = copy.deepcopy(self.output_cell.forward_func.weight.data)
        for idx in reversed(range(len(self.hidden_cells))):
            self.hidden_cells[idx].feedback_func.weight.data = copy.deepcopy(feedback_weight.t())
            if idx > 0:
                feedback_weight = torch.matmul(feedback_weight, self.hidden_cells[idx].forward_func.weight.data)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.error_func = error_func
        self.device = device
        self.cos_func = nn.CosineSimilarity(dim=0)

    def train_online(self, spike_data, label_one_hot, soft_error_step):
        """
        Train SNN with online learning

        Args:
            spike_data (Tensor): spike data input (batch_size, input_dim, spike_ts)
            label_one_hot (Tensor): one hot vector of label
            soft_error_step (int): soft start step for error feedback

        Returns:
            predict_label: predict labels
            hidden_forward_states: list of hidden forward states
            hidden_feedback_states: list of hidden feedback states
            out_forward_state: output forward state
            out_feedback_state: output feedback state
            feedback_step: number of steps for feedback simulation

        """
        batch_size = spike_data.shape[0]
        spike_ts = spike_data.shape[-1]
        if len(spike_data.shape) > 3:
            spike_data = spike_data.view(batch_size, self.input_dim, spike_ts)

        # Init Hidden Layer Cell States
        hidden_forward_states, hidden_feedback_states = [], []
        for cell in self.hidden_cells:
            forward_state, feedback_state = cell.train_reset_state(batch_size, self.device)
            hidden_forward_states.append(forward_state)
            hidden_feedback_states.append(feedback_state)

        # Init Output Layer Cell State
        out_forward_state, out_feedback_state = self.output_cell.train_reset_state(batch_size, self.device)

        # Start online simulation of the network
        output = torch.zeros([batch_size, self.output_cell.output_dim], device=self.device)
        feedback_step = 0
        for tt in range(spike_ts):
            input_spike = spike_data[:, :, tt]
            for idx, cell in enumerate(self.hidden_cells, 0):
                input_spike, hidden_forward_states[idx] = cell.train_forward_step(input_spike,
                                                                                  hidden_forward_states[idx])
            out_spike, out_forward_state = self.output_cell.train_forward_step(input_spike, out_forward_state)
            output = output + out_spike

            # Start feedback simulation after a soft start
            if tt >= soft_error_step:
                error = self.error_func(output, label_one_hot)
                error_pos = copy.deepcopy(error)
                error_pos[error_pos < 0] = 0
                error_neg = -copy.deepcopy(error)
                error_neg[error_neg < 0] = 0

                pos_spike, neg_spike, out_feedback_state = self.output_cell.train_feedback_step(
                    error_pos, error_neg, out_feedback_state)
                for idx in reversed(range(len(self.hidden_cells))):
                    hidden_feedback_states[idx] = self.hidden_cells[idx].train_feedback_step(
                        pos_spike, neg_spike, hidden_feedback_states[idx])
                feedback_step += 1

        # Predict label
        predict_label = torch.argmax(output, 1)

        return predict_label, hidden_forward_states, hidden_feedback_states, out_forward_state, out_feedback_state, feedback_step

    def train_update_parameter(self, hidden_forward_states, hidden_feedback_states,
                               out_forward_state, out_feedback_state, feedback_step, lr):
        """
        Update parameter of the SNN

        Args:
            hidden_forward_states (list): list of hidden forward states
            hidden_feedback_states (list): list of hidden feedback states
            out_forward_state (tuple): output forward state
            out_feedback_state (tuple): output feedback state
            feedback_step (int): number of steps for feedback simulation
            lr (float): learning rate

        """
        # Update Hidden Layer weight and bias
        for idx, cell in enumerate(self.hidden_cells, 0):
            trace_dw, trace_db = hidden_forward_states[idx][3], hidden_forward_states[idx][5]
            error_volt = hidden_feedback_states[idx]
            cell.train_update_parameter_sgd((error_volt, feedback_step, trace_dw, trace_db), lr)

        # Update Output Layer weight and bias
        trace_dw, trace_db = out_forward_state[3], out_forward_state[5]
        error_volt = out_feedback_state[4]
        self.output_cell.train_update_parameter_sgd((error_volt, feedback_step, trace_dw, trace_db), lr)

    def sleep_feedback_update(self, batch_size, spike_ts, oja_power, lr):
        """
        Sleep phase for feedback weight update using spike-based weight mirror

        Args:
            batch_size (int): batch size
            spike_ts (int): spike timesteps
            oja_power (float): oja power factor for oja decay
            lr (float): learning rate

        """
        noise_pos = torch.rand(1)[0]
        noise_neg = noise_pos

        for idx in reversed(range(len(self.hidden_cells))):
            # Generate Poisson Positive and Negative input spikes for this hidden layer
            hidden_output_dim = self.hidden_cells[idx].output_dim
            poisson_spike_pos = Bernoulli(torch.full_like(torch.zeros(batch_size, hidden_output_dim, spike_ts,
                                                                      device=self.device), noise_pos)).sample()
            poisson_spike_neg = Bernoulli(torch.full_like(torch.zeros(batch_size, hidden_output_dim, spike_ts,
                                                                      device=self.device), noise_neg)).sample()

            # Init Hidden Layer Cell States
            hidden_forward_states = []
            for ii in range(idx+1, len(self.hidden_cells)):
                forward_state = self.hidden_cells[ii].test_reset_state(batch_size, self.device)
                hidden_forward_states.append(forward_state)

            # Init Output Layer Cell State
            out_forward_state = self.output_cell.test_reset_state(batch_size, self.device)

            # Init Hidden Layer Spike Trace and Output Spike Trace
            hidden_spike_trace = torch.zeros(batch_size, hidden_output_dim, device=self.device)
            output_spike_trace = torch.zeros(batch_size, self.output_dim, device=self.device)

            # Start Sleeping for this Hidden Layer
            for tt in range(spike_ts):
                input_spike_pos = poisson_spike_pos[:, :, tt]
                input_spike_neg = poisson_spike_neg[:, :, tt]
                hidden_spike_trace = hidden_spike_trace + input_spike_pos - input_spike_neg
                if len(hidden_forward_states) == 0:
                    spike_output, out_forward_state = self.output_cell.sleep_forward_step(input_spike_pos,
                                                                                          input_spike_neg,
                                                                                          out_forward_state)
                else:
                    input_spike, hidden_forward_states[0] = self.hidden_cells[idx+1].sleep_forward_step(input_spike_pos,
                                                                                                        input_spike_neg,
                                                                                                        hidden_forward_states[0])
                    for ii in range(1, len(hidden_forward_states)):
                        input_spike, hidden_forward_states[ii] = self.hidden_cells[idx+ii+1].test_forward_step(input_spike,
                                                                                                               hidden_forward_states[ii])
                    spike_output, out_forward_state = self.output_cell.test_forward_step(input_spike, out_forward_state)
                output_spike_trace = output_spike_trace + spike_output

            # Compute Correlation for feedback weight update
            corr_batch_sum = torch.matmul(hidden_spike_trace.t(), output_spike_trace)

            # Compute Decay base on Oja's Rule
            oja_decay = torch.mul(torch.mean(torch.pow(output_spike_trace, oja_power), axis=0),
                                  self.hidden_cells[idx].feedback_func.weight.data)

            # Update Feedback Weights for this Hidden Layer
            self.hidden_cells[idx].feedback_func.weight.data += lr * (corr_batch_sum - oja_decay)

    def test(self, spike_data, batch_size):
        # For debugging
        print("spike_data.shape:", spike_data.shape)
        print("spike_data.numel():", spike_data.numel())

        # Determine input_dim dynamically
        channels = spike_data.shape[2]
        height = spike_data.shape[3]
        width = spike_data.shape[4]
        input_dim = channels * height * width

        # Time dimension
        spike_ts = spike_data.shape[0]

        # Check if we need to permute: from [time, batch, channels, H, W] to [batch, channels, H, W, time]
        if spike_data.shape != (batch_size, input_dim, spike_ts):
            spike_data = spike_data.permute(1, 2, 3, 4, 0)  # [batch, channels, H, W, time]
            spike_data = spike_data.reshape(batch_size, input_dim, spike_ts)

        # Check if the shapes don't match
        expected_elements = batch_size * input_dim * spike_ts
        if spike_data.numel() != expected_elements:
            raise RuntimeError(f"Reshape mismatch: {spike_data.numel()} vs expected {expected_elements}")

        # For debugging
        print("spike_data.shape:", spike_data.shape)
        print("target reshape:", batch_size, input_dim, spike_ts)

        # Init hidden states
        hidden_forward_states = [
            cell.test_reset_state(batch_size, self.device)
            for cell in self.hidden_cells
        ]
        out_forward_state = self.output_cell.test_reset_state(batch_size, self.device)

        # Record spikes over time
        spk_rec = torch.zeros([batch_size, self.output_cell.output_dim, spike_ts], device=self.device)

        for tt in range(spike_ts):
            input_spike = spike_data[:, :, tt]
            for idx, cell in enumerate(self.hidden_cells):
                input_spike, hidden_forward_states[idx] = cell.test_forward_step(input_spike, hidden_forward_states[idx])
            out_spike, out_forward_state = self.output_cell.test_forward_step(input_spike, out_forward_state)
            spk_rec[:, :, tt] = out_spike

        # Sum spikes over time
        spk_count = spk_rec.sum(dim=2)

        print("Final spk_count shape:", spk_count.shape)

        predict_label = torch.argmax(spk_count, dim=1)
        return spk_count, predict_label




    def compute_feedback_angle_ratio(self):
        """
        Compute angle and magnitude ratio between feedback connection and forward connection for each layer

        Returns:
            angle_list: list of angle (from lower hidden layer to higher hidden layer)
            ratio_list: list of magnitude ratio (from lower hidden layer to higher hidden layer)

        """
        angle_list, ratio_list = [], []
        forward_weight = copy.deepcopy(self.output_cell.forward_func.weight.data)
        for idx in reversed(range(len(self.hidden_cells))):
            feedback_weight = copy.deepcopy(self.hidden_cells[idx].feedback_func.weight.data)
            angle, ratio = self.compute_angle_ratio_between_weight_matrix(feedback_weight,
                                                                          copy.deepcopy(forward_weight.t()))
            angle_list.append(angle)
            ratio_list.append(ratio)
            if idx > 0:
                forward_weight = torch.matmul(forward_weight, self.hidden_cells[idx].forward_func.weight.data)

        return angle_list, ratio_list

    def compute_angle_ratio_between_weight_matrix(self, weight1, weight2):
        """
        Compute angle and magnitude ratio between two weight matrix

        Args:
            weight1 (Tensor): weight matrix 1
            weight2 (Tensor): weight matrix 2

        Returns:
            angle: angle between two weight matrix
            ratio: magnitude ratio between two weight matrix

        """
        flatten_weight1 = torch.flatten(weight1)
        flatten_weight2 = torch.flatten(weight2)
        weight1_norm = torch.norm(flatten_weight1)
        weight2_norm = torch.norm(flatten_weight2)
        ratio = (weight1_norm / weight2_norm).to('cpu').item()

        weight_cos = self.cos_func(flatten_weight1, flatten_weight2)
        angle = (180. / math.pi) * torch.acos(weight_cos).to('cpu').item()

        return angle, ratio

    def forward(self, x):
        print("Checking what x.shape[1] is", x.shape[1])
        return self.test(x, x.shape[1])