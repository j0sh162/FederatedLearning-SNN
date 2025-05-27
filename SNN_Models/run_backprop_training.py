import pickle
import torch
import numpy as np
import torch.optim as optim
from nmnist_exp.nmnist_compact_dataset import NMNISTDataset
from nmnist_exp.training_backprop import stbp_snn_training
from backprop_snn.network_w_backprop import WrapBackpropNetworkWithSTBP

# Read N-MNIST data
snn_ts = 60
dt = 5
unlimited_mem = False
train_ds = NMNISTDataset('./data/nmnist_compact_train', snn_ts, dt, 60000, unlimited_mem=unlimited_mem)
test_ds = NMNISTDataset('./data/nmnist_compact_test', snn_ts, dt, 10000, unlimited_mem=unlimited_mem)
print("Dataset Read Finished ...")

# Define SNN parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_shape = 2 * 34 * 34
out_dim = 10
snn_param = {'hidden_layer': [(0.3, 0.3, 0.3, 1.0),
                              (0.3, 0.3, 0.3, 1.0)],
             'out_layer': (0.3, 0.3, 0.3, 1.0)}

# Define Training parameters
val_size = 10000
train_batch_size = 128
test_batch_size = 256
epoch = 100
lr = 1.0e-4
# lr = 0.9e-2  # learning rate for SGD optimizer

# Define SNN and start training
hidden_dim_list = [500, 100]
seed_list = [0, 5, 10, 15, 20]

for seed in seed_list:
    torch.manual_seed(seed)
    np.random.seed(seed)
    session_name = "stbp_snn_adam_seed_" + str(seed)

    stbp_snn = WrapBackpropNetworkWithSTBP(in_shape, out_dim, hidden_dim_list, snn_param, device)
    optimizer = optim.Adam(stbp_snn.parameters(), lr=lr)
    # optimizer = optim.SGD(stbp_snn.parameters(), lr=lr)
    train_loss, val_acc, test_acc = stbp_snn_training(stbp_snn, optimizer, train_ds, test_ds, device, session_name,
                                                      validation_size=val_size,
                                                      batch_size=train_batch_size,
                                                      test_batch_size=test_batch_size, epoch=epoch)

    pickle.dump(train_loss, open("./save_models/" + session_name + "/train_loss_list.p", "wb+"))
    pickle.dump(val_acc, open("./save_models/" + session_name + "/val_accuracy_list.p", "wb+"))
    pickle.dump(test_acc, open("./save_models/" + session_name + "/test_accuracy_list.p", "wb+"))
