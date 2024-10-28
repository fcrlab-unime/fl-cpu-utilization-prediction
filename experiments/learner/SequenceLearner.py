import time
import torch
import logging
import numpy as np
import pandas as pd
from typing import Any
from typing import List
from flwr.common import Context
from flwr.client import Client, ClientApp
from flwr.simulation import run_simulation
from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from .client import FlowerClient
from collections import OrderedDict

class SequenceLearner:

    __instance = None

    def __new__(cls):
        if not cls.__instance:
            cls.__instance = super(SequenceLearner, cls).__new__(cls)
        return cls.__instance
     
    def __init__(self) -> None:
        self.__history = self.__get_empty_history()

    def __get_empty_history(self):
        return {
                "train": {
                    "mse": list(),
                    "rmse": list(),
                    "epoch_time": list()
                },
                "test": {
                    "mse": list(),
                    "rmse": list(),
                    "mae": list()
                },
                "predict": {
                    "y_test": '',
                    "y_pred": ''
                }
            }


    def build_dataset_name(self, dataset_name: str):
        self.__dataset_name = dataset_name
        return self.__instance
    
    def build_model(self, model: Any):
        self.__model = model
        return self.__instance
    
    def build_features_train(self, features: torch.Tensor):
        self.__features_train = features
        return self.__instance
    
    def build_targets_train(self, targets: torch.Tensor):
        self.__targets_train = targets
        return self.__instance

    def build_features_test(self, features: torch.Tensor):
        self.__features_test = features
        return self.__instance
    
    def build_targets_test(self, targets: torch.Tensor):
        self.__targets_test = targets
        return self.__instance
    
    def build_optimizer(self, optimizer: Any):
        self.__optimizer = optimizer
        return self.__instance
    
    def build_num_epoch(self, num_epochs: int):
        self.__num_epochs = num_epochs
        return self.__instance
    
    def build_num_rounds(self, num_rounds: int):
        self.__num_rounds = num_rounds
        return self.__instance
    
    def build_batch_size(self, batch_size: int):
        self.__batch_size = batch_size
        return self.__instance

    def __build_aggregation_strategy(self):
        self.__strategy = FedAvg(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
            min_fit_clients=3,  # Never sample less than 10 clients for training
            min_evaluate_clients=3,  # Never sample less than 5 clients for evaluation
            min_available_clients=3,  # Wait until all 10 clients are available
        )
        return 
        
    def build(self):
        self.__build_aggregation_strategy()
        return self.__instance
    

    def __client_fn(self, context: Context) -> Client:
        print(context.node_config)
        partition_id = context.node_config["partition-id"]
        logging.info("partition", partition_id)
        features_train = self.__features_train[partition_id]
        features_test = self.__features_test[partition_id]
        return FlowerClient(
            self.__model, features_train, features_test, self.__get_parameters, self.__set_parameters, self.train, self.__evaluate_one_epoch
        ).to_client()
    
    def __server_fn(self, context: Context) -> ServerAppComponents:
        config = ServerConfig(num_rounds=self.__num_rounds)
        return ServerAppComponents(strategy=self.__strategy, config=config)
    
    def start(self):
        logging.info("Running the Flower simulation ...")
        client = ClientApp(client_fn = self.__client_fn)
        logging.info("Defined the ClientApp")
        server = ServerApp(server_fn=self.__server_fn)
        logging.info("Defined the ServerApp")
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

        run_simulation(
            server_app=server,
            client_app=client,
            num_supernodes=3,
            backend_config=backend_config,
        )
    

    def __set_parameters(self, model, parameters: List[np.ndarray]):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def __get_parameters(self, model) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def train(self, evaluate: bool = False) -> None:
        logging.info("Sono su TRAIN")
        for epoch in range(self.__num_epochs):
            metrics = self.__train_one_epoch(evaluate)
            if evaluate:
                epoch_mse_train, epoch_rmse_train, epoch_time, epoch_mse_test, epoch_rmse_test, epoch_mae_test = metrics
            else:
                epoch_mse_train, epoch_rmse_train, epoch_time = metrics

            if epoch % 10 == 0:
                logging.info("Epoch: %d, train RMSE: %1.5f in %.2f s" % (epoch, epoch_rmse_train, epoch_time))
            
            self.__history['train']['mse'].append(epoch_mse_train)
            self.__history['train']['rmse'].append(epoch_rmse_train)
            self.__history['train']['epoch_time'].append(epoch_time)
            if evaluate:
                self.__history['test']['mse'].append(epoch_mse_test)
                self.__history['test']['rmse'].append(epoch_rmse_test)
                self.__history['test']['mae'].append(epoch_mae_test)
        
        return self.__history

    def __train_one_epoch(self, evaluate: bool = False) -> tuple[float, float, float]:
        loss_function = torch.nn.MSELoss()
        running_mse_train = 0

        self.__model.train()
        epoch_time_start = time.time()
        for _, i in enumerate(range(0, len(self.__features_train) - 1, self.__batch_size)):
            features_train_batch = self.__get_batch(self.__features_train, i)
            targets_train_batch = self.__get_batch(self.__targets_train, i)

            predicted_features_train_batch = self.__model.forward(features_train_batch)
            # calculate the gradient, manually setting to 0
            self.__optimizer.zero_grad()

            loss = loss_function(predicted_features_train_batch, targets_train_batch)
            running_mse_train += loss.item()

            loss.backward() # calculates the loss of the loss function
            self.__optimizer.step() # improve from loss, i.e backprop
        epoch_time_end = time.time()
        epoch_time = epoch_time_end-epoch_time_start

        epoch_mse_train = running_mse_train / len(self.__features_train)
        epoch_rmse_train = np.sqrt(epoch_mse_train)

        if evaluate:
            epoch_mse_test, epoch_rmse_test, epoch_mae_test = self.__evaluate_one_epoch()
            return epoch_mse_train, epoch_rmse_train, epoch_time, epoch_mse_test, epoch_rmse_test, epoch_mae_test

        return epoch_mse_train, epoch_rmse_train, epoch_time

    def __get_batch(self, data: torch.Tensor, i) -> tuple:
        sequence_lenght = min(self.__batch_size, len(data) - 1 - i)
        data = data[i:i+sequence_lenght]
        tensors = [item for item in data]
        return torch.stack(tensors)
    
    def __evaluate_one_epoch(self) -> tuple[float, float, float]:
        self.__model.eval()
        with torch.no_grad():
            running_mse_test = 0
            running_mae_test = 0
            for _, i in enumerate(range(0, len(self.__features_test) - 1, self.__batch_size)):
                features_test_batch = self.__get_batch(self.__features_test, i)
                targets_test_batch = self.__get_batch(self.__targets_test, i)

                # compute the MSE
                loss_function = torch.nn.MSELoss()
                predicted_targets_batch = self.__model(features_test_batch)
                loss = loss_function(predicted_targets_batch, targets_test_batch)
                running_mse_test += loss.item()

                # compute the MAE
                loss_function = torch.nn.L1Loss()
                predicted_targets_batch = self.__model(features_test_batch)
                loss = loss_function(predicted_targets_batch, targets_test_batch)
                running_mae_test += loss.item()
        
        epoch_mse_test = running_mse_test / len(self.__features_test)
        epoch_rmse_test = np.sqrt(epoch_mse_test)
        epoch_mae_test = running_mae_test / len(self.__features_test)
        
        return epoch_mse_test, epoch_rmse_test, epoch_mae_test
    
    
    def predict(self, normalizer, y_test):
        with torch.no_grad():
            y_pred = self.__model(self.__features_test)
            # nsamples, nx, ny = y_pred.shape
            # y_pred = y_pred.reshape((nsamples,nx*ny))
            y_pred = normalizer.inverse_transform(y_pred.cpu())
        y_test = normalizer.inverse_transform(y_test)

        y_test_plt = pd.DataFrame(y_test, columns=['min_cpu','max_cpu','avg_cpu'])
        y_pred_plt = pd.DataFrame(y_pred, columns=['min_cpu','max_cpu','avg_cpu'])

        return y_test_plt, y_pred_plt