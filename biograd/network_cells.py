import torch
import torch.nn as nn


class OnlineHiddenCell(nn.Module):
    """ Online Fully-Connected Spiking Neuron Cell for Hidden Layers """

    def __init__(self, forward_func, feedback_func, neuron_param, input_dim, output_dim):
        """

        Args:
            forward_func (Torch Function): Pre-synaptic function for forward connection
            feedback_func (Torch Function): Feedback function for feedback connection
            neuron_param (tuple): LIF neuron parameters
            input_dim (int): input dimension
            output_dim (int): output dimension
        """
        super().__init__()
        self.forward_func = forward_func
        self.feedback_func = feedback_func
        self.vdecay, self.vth, self.grad_win, self.grad_amp = neuron_param
        self.input_dim = input_dim
        self.output_dim = output_dim

    def train_reset_state(self, batch_size, device):
        """
        At start of training, reset all states within the neuron

        Args:
            batch_size (int): batch size
            device (device): device

        Returns:
            forward_state: forward neuron states
            feedback_state: feedback neuron state

        """
        # Forward neuron states
        volt = torch.zeros([batch_size, self.output_dim], device=device)  # soma voltage
        spike = torch.zeros([batch_size, self.output_dim], device=device)  # soma spike
        trace_pre = torch.zeros([batch_size, self.output_dim, self.input_dim], device=device)  # pre-spike trace
        trace_dw = torch.zeros([batch_size, self.output_dim, self.input_dim],
                               device=device)  # gradient trace for weight
        trace_bias = torch.zeros([batch_size, self.output_dim], device=device)  # bias-spike trace (spike all step)
        trace_db = torch.zeros([batch_size, self.output_dim], device=device)  # gradient trace for bias
        forward_state = (volt, spike, trace_pre, trace_dw, trace_bias, trace_db)

        # Feedback neuron states
        feedback_state = torch.zeros([batch_size, self.output_dim], device=device)  # error dendrite volt

        return forward_state, feedback_state

    def train_forward_step(self, spike_input, forward_state):
        """
        One step forward connection simulation for the neuron training

        Args:
            spike_input (Tensor): spike input from pre-synaptic input
            forward_state (tuple): forward neuron states

        Returns:
            spike_output: spike output to downstream layer
            forward_state: updated forward neuron states

        """
        volt, spike, trace_pre, trace_dw, trace_bias, trace_db = forward_state

        # Update neuron soma (LIF neuron)
        volt = self.vdecay * volt * (1. - spike) + self.forward_func(spike_input)
        spike_output = volt.gt(self.vth).float()

        # Update neuron traces
        volt_pseudo_grad = (abs(volt - self.vth) < self.grad_win).float() * self.grad_amp

        trace_pre = self.vdecay * trace_pre + spike_input.view(-1, 1, self.input_dim)
        # print("Pre Trace Max ", trace_pre.max(), end=" ")
        trace_dw = trace_dw + trace_pre * volt_pseudo_grad.view(-1, self.output_dim, 1)
        # print("Dw Trace Max ", trace_dw.max(), end=" ")
        trace_pre = trace_pre * (1 - spike_output - volt * volt_pseudo_grad).view(-1, self.output_dim, 1)

        trace_bias = self.vdecay * trace_bias + 1.
        # print("Bias Trace Max ", trace_bias.max(), end=" ")
        trace_db = trace_db + trace_bias * volt_pseudo_grad
        # print("Db Trace Max ", trace_db.max())
        trace_bias = trace_bias * (1 - spike_output - volt * volt_pseudo_grad)

        return spike_output, (volt, spike_output, trace_pre, trace_dw, trace_bias, trace_db)

    def train_feedback_step(self, pos_spike_input, neg_spike_input, feedback_state):
        """
        One step feedback connection simulation for the neuron training

        Args:
            pos_spike_input (Tensor): spike input from downstream positive error neuron
            neg_spike_input (Tensor): spike input from downstream negative error neuron
            feedback_state (tuple): feedback neuron states

        Returns:
            feedback_state: updated feedback neuron states

        """
        # Update error dendrite
        error_dendrite_volt = feedback_state + (
                    self.feedback_func(pos_spike_input) - self.feedback_func(neg_spike_input))

        return error_dendrite_volt

    def train_update_parameter_sgd(self, update_state, lr):
        """
        Update parameter using vanilla SGD

        Args:
            update_state (tuple): neuron states used for update parameter
            lr (float): learning rate

        Returns:
            error: estimated error for hidden neurons by direct feedback connection

        """
        error_dendrite_volt, error_steps, trace_dw, trace_db = update_state
        error = error_dendrite_volt / error_steps
        mean_dw = torch.mean(error.view(-1, self.output_dim, 1) * trace_dw, 0)
        mean_db = torch.mean(error.view(-1, self.output_dim) * trace_db, 0)
        self.forward_func.weight.data -= lr * mean_dw
        self.forward_func.bias.data -= lr * mean_db

        return error

    def test_reset_state(self, batch_size, device):
        """
        At start of testing, reset all states within the neuron

        Args:
            batch_size (int): batch size
            device (device): device

        Returns:
            forward_state: forward neuron states

        """
        # Forward neuron states
        volt = torch.zeros([batch_size, self.output_dim], device=device)  # soma voltage
        spike = torch.zeros([batch_size, self.output_dim], device=device)  # soma spike
        forward_state = (volt, spike)

        return forward_state

    def test_forward_step(self, spike_input, forward_state):
        """
        One step forward connection simulation for the neuron (test only)

        Args:
            spike_input (Tensor): spike input from pre-synaptic input
            forward_state (tuple): forward neuron states

        Returns:
            spike_output: spike output to downstream layer
            forward_state: updated forward neuron states

        """
        volt, spike = forward_state

        # Update LIF neuron
        volt = self.vdecay * volt * (1. - spike) + self.forward_func(spike_input)
        spike_output = volt.gt(self.vth).float()

        return spike_output, (volt, spike_output)

    def sleep_forward_step(self, spike_input_pos, spike_input_neg, forward_state):
        """
        One step forward connection simulation for sleep phase of the neuron

        Args:
            spike_input_pos (Tensor): positive Poisson spike input
            spike_input_neg (Tensor): negative Poisson spike input
            forward_state (tuple): forward neuron states

        Returns:
            spike_output: spike output to downstream layer
            forward_state: updated forward neuron states

        """
        volt, spike = forward_state

        # Update LIF neuron
        volt = self.vdecay * volt * (1. - spike) + self.forward_func(spike_input_pos) - self.forward_func(spike_input_neg)
        spike_output = volt.gt(self.vth).float()

        return spike_output, (volt, spike_output)


class OnlineOutputCell(OnlineHiddenCell):
    """ Online Fully-Connected Spiking Neuron Cell for Output Layer (including error interneurons) """

    def __init__(self, forward_func, neuron_param, input_dim, output_dim):
        """

        Args:
            forward_func (Torch Function): Pre-synaptic function for forward
            neuron_param (tuple): LIF neuron and feedback parameters
            input_dim (int): input dimension
            output_dim (int): output dimension
        """
        super(OnlineOutputCell, self).__init__(forward_func,
                                               nn.Identity(),
                                               neuron_param,
                                               input_dim,
                                               output_dim)

    def train_reset_state(self, batch_size, device):
        """
        At start of training, reset all states within the neuron

        Args:
            batch_size (int): batch size
            device (device): device

        Returns:
            forward_state: forward neuron states
            feedback_state: feedback neuron states

        """
        # Forward neuron states
        volt = torch.zeros([batch_size, self.output_dim], device=device)  # soma voltage
        spike = torch.zeros([batch_size, self.output_dim], device=device)  # soma spike
        trace_pre = torch.zeros([batch_size, self.output_dim, self.input_dim], device=device)  # pre-spike trace
        trace_dw = torch.zeros([batch_size, self.output_dim, self.input_dim],
                               device=device)  # gradient trace for weight
        trace_bias = torch.zeros([batch_size, self.output_dim], device=device)  # bias-spike trace (spike all step)
        trace_db = torch.zeros([batch_size, self.output_dim], device=device)  # gradient trace for bias
        forward_state = (volt, spike, trace_pre, trace_dw, trace_bias, trace_db)

        # Feedback neuron states
        error_pos_volt = torch.zeros([batch_size, self.output_dim], device=device)  # error pos neuron volt
        error_neg_volt = torch.zeros([batch_size, self.output_dim], device=device)  # error neg neuron volt
        error_pos_spike = torch.zeros([batch_size, self.output_dim], device=device)  # error pos neuron spike
        error_neg_spike = torch.zeros([batch_size, self.output_dim], device=device)  # error neg neuron spike
        error_dendrite_volt = torch.zeros([batch_size, self.output_dim], device=device)  # error dendrite volt
        feedback_state = (error_pos_volt, error_neg_volt, error_pos_spike, error_neg_spike, error_dendrite_volt)

        return forward_state, feedback_state

    def train_feedback_step(self, pos_input, neg_input, feedback_state):
        """
        One step feedback simulation for the neuron

        Args:
            pos_input (Tensor): current input from error computation
            neg_input (Tensor): current input from error computation
            feedback_state (tuple): feedback neuron states

        Returns:
            pos_spike_output: spike output to upstream positive error neuron
            neg_spike_output: spike output to upstream negative error neuron
            feedback_state: updated feedback neuron states

        """
        error_pos_volt, error_neg_volt, error_pos_spike, error_neg_spike, error_dendrite_volt = feedback_state

        # Update error neurons (IF neurons with soft reset)
        error_neuron_psp = self.feedback_func(pos_input) - self.feedback_func(neg_input)

        error_pos_volt = error_pos_volt - error_pos_spike + error_neuron_psp
        pos_spike_output = error_pos_volt.gt(1.0).float()

        error_neg_volt = error_neg_volt - error_neg_spike - error_neuron_psp
        neg_spike_output = error_neg_volt.gt(1.0).float()

        # Update error dendrite
        error_dendrite_volt = error_dendrite_volt + (pos_spike_output - neg_spike_output)

        return pos_spike_output, neg_spike_output, (
        error_pos_volt, error_neg_volt, pos_spike_output, neg_spike_output, error_dendrite_volt)
