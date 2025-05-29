import torch
import torch.nn as nn
from snntorch import functional as SF
import pickle
import torch
import time
import numpy as np
from SNN_Models.nmnist_dataset import NMNISTDataset
from SNN_Models.training_biograd import biograd_snn_training
from SNN_Models.online_error_functions import cross_entropy_loss_error_function
from torch.utils.data import DataLoader, sampler

def train(network, train_loader, optimizer, epochs, device: str,
                         validation_size=10000, batch_size=1, sleep_batch_size=1,
                         test_batch_size=1, epoch=100,
                         save_epoch=1, lr=1e-3,
                         sleep_oja_power=2, sleep_lr=1e-3, sleep_spike_ts=50, soft_error_step=19):
    print("c")
    """
    BioGrad SNN training with sleep

    Args:
        network (SNN): Online learning SNN
        train_dataset (dataset): training dataset
        test_dataset (dataset): test dataset
        sleep_spike_ts (int): spike timestep for sleep
        device (device):device
        soft_error_step (int): soft start error step for feedback simulation
        session_name (str): name of the training session
        validation_size (int): size of validation set
        batch_size (int): batch size for training
        sleep_batch_size (int): batch size for sleep
        test_batch_size (int): batch size for testing
        epoch (int): number of epoches
        save_epoch (int): every number of epoch to save model
        lr (float): learning rate
        sleep_oja_power (float): oja power for oja decay for sleep
        sleep_lr (float): learning rate for sleep

    Returns:
        train_accuracy_list: list of training accuracy for each epoch
        val_accuracy_list: list of validation accuracy for each epoch
        test_accuracy_list: list of test accuracy for each epoch
        feedback_angle: list of feedback angle of each hidden layer
        feedback_ratio: list of feedback ratio of each hidden layer
    """

    # Train, validation, and test dataloader

    for data in train_loader:
        images, labels = data
        print(f"DEBUG — DataLoader batch size: {images.shape[0]}")

    # Number of samples in train, validation, and test dataset
    train_num = len(train_loader.dataset)

    # List for save accuracy
    train_accuracy_list, val_accuracy_list, test_accuracy_list = [], [], []

    # Compute init angle and ratio between feedback weight and forward weight
    feedback_angle, feedback_ratio = [], []
    angle_list, ratio_list = network.compute_feedback_angle_ratio()
    feedback_angle.append(angle_list)
    feedback_ratio.append(ratio_list)

    # Start training
    with torch.no_grad():
        for ee in range(epoch):
            # Training
            train_correct = 0
            train_start = time.time()
            for i, data in enumerate(train_loader):
                print("Batch", str(i) + "/" + str(len(train_loader)))
                event_img, labels = data
                labels_one_hot = nn.functional.one_hot(labels, num_classes=10).float()
                event_img, labels, labels_one_hot = event_img.to(device), labels.to(device), labels_one_hot.to(device)
                predict_label, hid_fwd_states, hid_fb_states, out_fwb_state, out_fb_state, fb_step = network.train_online(
                    event_img, labels_one_hot, soft_error_step)
                network.train_update_parameter(hid_fwd_states, hid_fb_states, out_fwb_state, out_fb_state, fb_step, lr)
                train_correct += ((predict_label == labels).sum().to("cpu")).item()

                # Put network to sleep for feedback training
                network.sleep_feedback_update(sleep_batch_size, sleep_spike_ts, sleep_oja_power, sleep_lr)

                # Compute angle and ratio between feedback weight and forward weight after each update
                angle_list, ratio_list = network.compute_feedback_angle_ratio()
                feedback_angle.append(angle_list)
                feedback_ratio.append(ratio_list)

            train_end = time.time()
            train_accuracy_list.append(train_correct / train_num)
            print("Epoch %d Training Accuracy %.4f" % (ee, train_accuracy_list[-1]), end=" ")
            """
            # Validation
            val_correct = 0
            val_start = time.time()
            for data in val_dataloader:
                event_img, labels = data
                event_img, labels = event_img.to(device), labels.to(device)
                predict_label = network.test(event_img)
                val_correct += ((predict_label == labels).sum().to("cpu")).item()
            val_end = time.time()
            val_accuracy_list.append(val_correct / val_num)
            tf_writer.add_scalar('nmnist_exp/val_accuracy', val_accuracy_list[-1], ee)
            print("Validate Accuracy %.4f" % val_accuracy_list[-1], end=" ")
            """

    print("End Training")
    #return train_accuracy_list, val_accuracy_list, test_accuracy_list, feedback_angle, feedback_ratio


def test(net, testloader, device: str):
    print("d")
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for i, data in enumerate(testloader):
            print("Batch", str(i) + "/" + str(len(testloader)))
            images, labels = data[0].to(device), data[1].to(device)
            outputs, _ = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
