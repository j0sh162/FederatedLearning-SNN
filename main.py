# Main class to run federated learning
import pickle
from pathlib import Path

import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich import print
from sympy import evaluate

from FL.client import generate_client_fn
from FL.server import get_evaluate_fn, get_on_fit_config
from Training import dataset

from SNN_Models.nmnist_dataset import NMNISTDataset
from SNN_Models.training_biograd import biograd_snn_training
from SNN_Models.online_error_functions import cross_entropy_loss_error_function
from torch.utils.data import DataLoader, sampler


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Parse config and get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare dataset
    snn_ts = 60
    dt = 5
    unlimited_mem = False
    validation_size = 10000
    train_dataset = NMNISTDataset('./data/NMNIST/Train', snn_ts, dt)
    train_idx = [idx for idx in range(len(train_dataset) - validation_size)]
    val_idx = [(idx + len(train_idx)) for idx in range(validation_size)]
    train_sampler = sampler.SubsetRandomSampler(train_idx)
    val_sampler = sampler.SubsetRandomSampler(val_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler,
                                  shuffle=False, num_workers=4)

    snn_ts = 60
    dt = 5
    unlimited_mem = False
    test_dataset = NMNISTDataset('./data/NMNIST/Test', snn_ts, dt)
    test_idx = [idx for idx in range(len(test_dataset))]
    #val_idx = [(idx + len(train_idx)) for idx in range(validation_size)]
    test_sampler = sampler.SubsetRandomSampler(test_idx)
    val_dataloader = DataLoader(train_dataset, batch_size=1, sampler=val_sampler,
                                shuffle=False, num_workers=4)
    #val_sampler = sampler.SubsetRandomSampler(val_idx)
    test_dataloader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler,
                                  shuffle=False, num_workers=4)

    # 3. Define clients - allows to initialize clients
    client_fn = generate_client_fn(train_dataloader, val_dataloader, cfg.model)
    print(
        "cfg num_rounds: ", cfg.fl.num_rounds, "cfg num classes", cfg.client.num_classes
    )

    # 4. Define Strategy
    # strategy = fl.server.strategy.FedAvg(fraction_fit=0.00001,
    #                                      min_fit_clients=cfg.client.num_clients_per_round_fit,
    #                                      fraction_evaluate=0.00001,
    #                                      min_evaluate_clients=cfg.client.num_clients_per_round_evaluate,
    #                                      min_available_clients=cfg.fl.num_clients,
    #                                      on_fit_config_fn=get_on_fit_config(cfg.client),
    #                                      evaluate_fn = get_evaluate_fn(cfg.client.num_classes,testLoader)) #At the end of aggregation we obtain new global model and evaluate it
    strategy = instantiate(
        cfg.strategy, evaluate_fn=get_evaluate_fn(cfg.model, test_dataloader)
    )
    # 5. Run your simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.fl.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.fl.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 6,  # was 2
            "num_gpus": 1,
        },  # run client concurrently on gpu 0.25 = 4 clients concurrently
    )
    # 6. Save results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "results.pkl"
    results = {"history": history}  # save anything else needed
    with open(results_path, "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
