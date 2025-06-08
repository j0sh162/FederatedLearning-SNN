#Main class to run federated learning
from hydra.utils import instantiate
from sympy import evaluate
from Training import dataset
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
from FL.client import generate_client_fn
import flwr as fl
from FL.server import get_on_fit_config, get_evaluate_fn
from hydra.core.hydra_config import HydraConfig
from pathlib import Path



@hydra.main(config_path="conf",config_name="config",version_base=None)
def main(cfg: DictConfig):
    #1. Parse config and get experiment output dir ()
    print(OmegaConf.to_yaml(cfg))

    #2. Prepare dataset
    dataset_name = cfg.dataset.name
    dataset_path = cfg.datasets[dataset_name].path
    trainLoaders, validationLoaders, testLoader = dataset.load_dataset(dataset_name,dataset_path,cfg.fl.num_clients,cfg.fl.batch_size,0.1)
    print(len(trainLoaders),len(trainLoaders[0].dataset))

    #3. Define clients - allows to initialize clients
    client_fn=generate_client_fn(trainLoaders,validationLoaders,cfg.model)
    #print("cfg num_rounds: ",cfg.fl.num_rounds,"cfg num classes",cfg.client.num_classes)

    #4. Define Strategy (The aggregation strategy)
    strategy = instantiate(cfg.strategy,
                           evaluate_fn=get_evaluate_fn(cfg.model, testLoader)
                           )
    print("strategy:",cfg.strategy)
    #5. Run your simulation
    history =  fl.simulation.start_simulation(client_fn=client_fn, #spawns client
                                              num_clients=cfg.fl.num_clients, #total num of clients
                                              config=fl.server.ServerConfig(num_rounds=cfg.fl.num_rounds), #how many rounds do we want to do?
                                              strategy=strategy, #what aggregation strategy we want to use
                                              #client_resources={'num_cpus': 2, 'num_gpus': 0}, #run client concurrently on gpu 0.25 = 4 clients concurrently
                                            )

    #6. Save results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / 'results.pkl'
    results = {'history': history} #save anything else needed
    with open(results_path, 'wb') as h:
        pickle.dump(results, h,protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == "__main__":
    main()