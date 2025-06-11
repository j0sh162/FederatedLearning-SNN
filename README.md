# FederatedLearning-SNN
This repository was the results of a Masters research project of Maastricht University. It is not a fully cleaned up package. The project is about training methods of Spiking Neural Networks (SNNs) and their implementation in a Federated Learning environment compared to a centralized training. The goal was to analyze the energy efficiency of the different training methods in federated learning compared to the centrally trained versions. Other research areas where how SNNs and non independent and identically distributed (iid) data affect each other with respect to the different aggregation methods in Federated Learning.

# Installation
To install the packages execute the following command:
```bash
pip install -e .
```

## Installation issues
We know there can be problems with installations on windows systems this fix might help:
For that install desktop development with C++ from using visual studio build tools, if needed: [https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)

# Run experiments
The experiments with federated learning are executed using the [Hydra package](https://hydra.cc/). The parameters of each experiment are defined using the config files in `conf/`. At the start of the program `conf/config.yaml` is loaded in, which also defines the starting values. You can look [here](https://hydra.cc/docs/patterns/configuring_experiments/) on how to configure the config files for each run.

An example call for running an experiment could be this:

```bash
python main.py model=SNN dataset.name=NMNIST ++seed=0 ++fl.num_rounds=1 ++fl.num_clients=10 ++client.num_clients_per_round_fit=10 ++client.num_clients_per_round_evaluate=10 ++client.lr=0.0002 ++client.local_epochs=1
```
Overwrites can be done using the `++` prefix.
