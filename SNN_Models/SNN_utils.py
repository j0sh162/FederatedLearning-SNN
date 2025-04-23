import torch
from snntorch import functional as SF


def train(net, train_loader, optimizer, epochs, device: str):
    net.train()
    net.to(device)
    num_batches = len(train_loader)
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    for epoch in range(epochs):
        for batch_number, (data, targets) in enumerate(iter(train_loader)):
            try:
                data = data.to(device)
                targets = targets.to(device)

                net.net.train()
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
                    )
                )
            except Exception as e:
                print("Error in training loop:", e)
                break


def test(net, testloader, device: str):
    criterion = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    acc_sum, loss_sum = 0.0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        i = 0
        for data in testloader:
            i += 1
            if i > 10:
                break

            images, labels = data[0].to(device), data[1].to(device)
            spk_rec = net(images)
            loss = criterion(spk_rec, labels)
            acc = SF.accuracy_rate(spk_rec, labels)
            loss_sum += loss.item()
            acc_sum += acc.item()
    loss = loss_sum / len(testloader.dataset)
    accuracy = acc_sum / len(testloader.dataset)
    return loss, accuracy
