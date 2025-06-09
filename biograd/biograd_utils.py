import time

import torch
from snntorch import functional as SF
from torch import nn
from tqdm import tqdm, trange

# Define Training parameters
soft_error_step = 19
sleep_spike_ts = 50
lr = 1.0e-3
sleep_oja_power = 2.0
sleep_lr = 1.0e-4 / 3.0


def train(net, train_loader, device: str, epochs, optimizer):
    feedback_angle, feedback_ratio = [], []
    angle_list, ratio_list = net.compute_feedback_angle_ratio()
    feedback_angle.append(angle_list)
    feedback_ratio.append(ratio_list)
    # Start training
    with torch.no_grad():
        for ee in trange(epochs, desc="Epoch"):
            # Training
            train_num = len(train_loader.dataset)
            train_correct = 0
            train_start = time.time()
            for data in tqdm(train_loader, desc="Train", leave=False):
                event_img, labels = data
                labels_one_hot = nn.functional.one_hot(labels, num_classes=10).float()
                event_img, labels, labels_one_hot = (
                    event_img.to(device),
                    labels.to(device),
                    labels_one_hot.to(device),
                )
                (
                    predict_label,
                    hid_fwd_states,
                    hid_fb_states,
                    out_fwb_state,
                    out_fb_state,
                    fb_step,
                ) = net.train_online(event_img, labels_one_hot, soft_error_step)
                net.train_update_parameter(
                    hid_fwd_states,
                    hid_fb_states,
                    out_fwb_state,
                    out_fb_state,
                    fb_step,
                    lr,
                )
                train_correct += ((predict_label == labels).sum().to("cpu")).item()

                # Put network to sleep for feedback training
                sleep_batch_size = event_img.size(0)
                net.sleep_feedback_update(
                    sleep_batch_size, sleep_spike_ts, sleep_oja_power, sleep_lr
                )

                # Compute angle and ratio between feedback weight and forward weight after each update
                angle_list, ratio_list = net.compute_feedback_angle_ratio()
                feedback_angle.append(angle_list)
                feedback_ratio.append(ratio_list)

            train_end = time.time()
            train_accuracy = train_correct / train_num
            print(
                "Epoch %d Training Accuracy %.4f Train time: %.1f"
                % (ee, train_accuracy, train_end - train_start),
                end=" ",
            )


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
        print(
            "Test Accuracy %.4f Test Time: %.1f"
            % (test_accuracy, test_end - test_start),
            end=" ",
        )
        # TODO implement loss function, might not work with snntorch function
        # loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
        return -1, test_accuracy
