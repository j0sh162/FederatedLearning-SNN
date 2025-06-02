# Main class to run federated learning
import pickle
from pathlib import Path

import flwr as fl
import hydra
from flwr.server.strategy import (
    DifferentialPrivacyClientSideFixedClipping,
    DifferentialPrivacyServerSideFixedClipping,
)
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich import print
from sympy import evaluate

from FL.client import generate_client_fn
from FL.server import get_evaluate_fn, get_on_fit_config
from Training import dataset


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Parse config and get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare dataset
    dataset_name = cfg.dataset.name
    dataset_path = cfg.datasets[dataset_name].path
    trainLoaders, validationLoaders, testLoader = dataset.load_dataset(
        dataset_name,
        dataset_path,
        cfg.fl.num_clients,
        cfg.datasets[dataset_name].batch_size,  # was originally cfg.fl.batch_size
        0.1,
    )
    print(len(trainLoaders), len(trainLoaders[0].dataset))

    # 3. Define clients - allows to initialize clients
    client_fn = generate_client_fn(trainLoaders, validationLoaders, cfg.model)
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
    base_strategy = instantiate(
        cfg.strategy, evaluate_fn=get_evaluate_fn(cfg.model, testLoader)
    )

    # Server-side central differential privacy
    if cfg.fl.differential_privacy:
        strategy = DifferentialPrivacyServerSideFixedClipping(
            base_strategy,
            cfg.fl.noise_multiplier,
            cfg.fl.clipping_norm,
            cfg.fl.num_sampled_clients,
        )
    else:
        strategy = base_strategy

    # 5. Run your simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.fl.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.fl.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 8,  # was 2
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
