import pickle

from rich import print

output = pickle.load(open(r"outputs\2025-06-07\15-35-08\results.pkl", "rb"))

print(output)
history = output["history"]
for metric_name, values in history.metrics_distributed.items():
    for round_num, value in values:
        print(f"distributed/{metric_name}", value, round_num)
for metric_name, values in history.metrics_centralized.items():
    for round_num, value in values:
        print(f"centralized/{metric_name}", value, round_num)
for round_num, loss in history.losses_distributed:
    print("distributed/loss", loss, round_num)
for round_num, loss in history.losses_centralized:
    print("centralized/loss", loss, round_num)

import torch

from SNN_Models.SNN import Net

model = Net()
model.load_state_dict(
    torch.load(open(r"outputs\2025-06-07\18-13-57\state_dict.pth", "rb"))
)
print(model.state_dict())
