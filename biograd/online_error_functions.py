import torch.nn as nn


def cross_entropy_loss_error_function(network_output, label_one_hot):
    """
    Compute output error signal (Here use CrossEntropyLoss)

    Args:
        network_output (Tensor): output of the SNN
        label_one_hot (Tensor): one hot vector of the label

    Returns:
        error: error vector

    """
    output_softmax = nn.functional.softmax(network_output, dim=-1)
    error = output_softmax - label_one_hot

    return error
