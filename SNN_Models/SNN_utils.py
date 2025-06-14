import torch
from flwr import common
from snntorch import functional as SF
from snntorch import surrogate, utils


def train(net, train_loader, optimizer, epochs, device: str):
    net.train()
    net.net.train()
    net.to(device)
    num_batches = len(train_loader)
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    for epoch in range(epochs):
        for batch_number, (data, targets) in enumerate(iter(train_loader)):
            try:
                data = data.to(device)
                targets = targets.to(device)

                spk_rec = net.forward(data)
                loss_val = loss_fn(spk_rec, targets)

                # Gradient calculation + weight update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                acc = SF.accuracy_rate(spk_rec, targets)
                print(
                    "Epoch {:02d} | Batch {:03d}/{:03d} | Loss {:05.2f} | Accuracy {:05.2f}%".format(
                        epoch, batch_number, num_batches, loss_val.item(), acc * 100
                    ),
                )
            except Exception as e:
                print("Error in training loop:", e)
                break


def test(net, testloader, device: str):
    criterion = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    acc_hist, loss_hist = [], []
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            spk_rec = net(images)
            loss = criterion(spk_rec, labels)
            acc = SF.accuracy_rate(spk_rec, labels)
            loss_hist.append(loss.item())
            acc_hist.append(acc.item())

    if len(loss_hist) == 0:
        return 0.0, 0.0
    loss = sum(loss_hist) / len(loss_hist)
    accuracy = sum(acc_hist) / len(acc_hist)
    return loss, accuracy
