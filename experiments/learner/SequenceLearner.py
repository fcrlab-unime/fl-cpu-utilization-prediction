import time
import torch
import logging
import numpy as np
import pandas as pd
from typing import Any, List, Tuple
from torchinfo import summary
from flwr.common import Context, Metrics
from flwr.client import Client, ClientApp
from flwr.simulation import run_simulation
from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from .client import FlowerClient
from collections import OrderedDict
import json

class SequenceLearner():

    __instance = None

    def __new__(cls):
        if not cls.__instance:
            cls.__instance = super(SequenceLearner, cls).__new__(cls)
        return cls.__instance
    
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
                    "mae": list(),
                    "r2": list()
                },
                "predict": {
                    "y_test": list(),
                    "y_pred": list()
                }
            }
    
    def __init__(self) -> None:
        self.__history = self.__get_empty_history()

    
    def build_model(self, model: Any):
        self.__model = model
        return self.__instance
    
    def build_optimizer(self, optimizer: Any):
        self.__optimizer = optimizer
        return self.__instance

    def build_dataset_name(self, dataset_name: str):
        self.__dataset_name = dataset_name
        return self.__instance
    
    def build_features_tensor_train(self, features: torch.Tensor):
        self.__features_tensor_train = features
        return self.__instance
    
    def build_targets_tensor_train(self, targets: torch.Tensor):
        self.__targets_tensor_train = targets
        return self.__instance

    def build_features_tensor_test(self, features: torch.Tensor):
        self.__features_tensor_test = features
        return self.__instance
    
    def build_targets_tensor_test(self, targets: torch.Tensor):
        self.__targets_tensor_test = targets
        return self.__instance
    
    def build_batch_size(self, batch_size: int):
        self.__batch_size = batch_size
        return self.__instance
    
    def build_lookback(self, lookback: int):
        self.__lookback = lookback
        return self.__instance
    
    def build_normalizer(self, normalizer):
        self.__normalizer = normalizer
        return self.__instance
    
    def build_num_epoch(self, num_epochs: int):
        self.__num_epochs = num_epochs
        return self.__instance
    
    def build_num_rounds(self, num_rounds: int):
        self.__num_rounds = num_rounds
        return self.__instance
    
    def build_num_clients(self, num_clients: int):
        self.__num_clients = num_clients
        return self.__instance
    
    def build_fraction_fit(self, fraction_fit: float):
        self.__fraction_fit = fraction_fit
        return self.__instance
    
    def build_fraction_evaluate(self, fraction_evaluate: float):
        self.__fraction_evaluate = fraction_evaluate
        return self.__instance
    
    def build_min_fit_clients(self, min_fit_clients: int):
        self.__min_fit_clients = min_fit_clients
        return self.__instance
    
    def build_min_evaluate_clients(self, min_evaluate_clients: int):
        self.__min_evaluate_clients = min_evaluate_clients
        return self.__instance
    
    def build_min_available_clients(self, min_available_clients: int):
        self.__min_available_clients = min_available_clients
        return self.__instance
    
    def build_history_fill_callback(self, history_fill_callback: Any):
        self.__history_fill_callback = history_fill_callback
        return self.__instance
    
    def build_predict_callback(self, predict_callback: Any):
        self.__predict_callback = predict_callback
        return self.__instance
    
    def build(self):
        return self.__instance
    

    def __set_parameters(self, model: Any, parameters: List[np.ndarray]):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        model.load_state_dict(state_dict, strict=True)

    def __get_parameters(self, model: Any) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in model.state_dict().items()]
    

    def __train_many_epochs(self,
                model: Any,
                optimizer: Any,
                features_train: torch.Tensor,
                targets_train: torch.Tensor,
                evaluate: bool = False
            ) -> None:
        for epoch in range(self.__num_epochs):
            metrics = self.__train_one_epoch(
                model,
                optimizer,
                features_train,
                targets_train,
                evaluate
            )
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

    def __train_one_epoch(self,
                model: Any,
                optimizer: Any,
                features_train: torch.Tensor,
                targets_train: torch.Tensor,
                evaluate: bool = False,
            ) -> tuple[float, float, float]:
        loss_function = torch.nn.MSELoss()
        running_mse_train = 0

        model.train()
        epoch_time_start = time.time()
        for _, i in enumerate(range(0, len(features_train) - 1, self.__batch_size)):
            features_train_batch = self.__get_batch(features_train, i)
            targets_train_batch = self.__get_batch(targets_train, i)

            predicted_features_train_batch = model.forward(features_train_batch)
            # calculate the gradient, manually setting to 0
            optimizer.zero_grad()

            loss = loss_function(predicted_features_train_batch, targets_train_batch)
            running_mse_train += loss.item()

            loss.backward() # calculates the loss of the loss function
            optimizer.step() # improve from loss, i.e backprop
        epoch_time_end = time.time()
        epoch_time = epoch_time_end-epoch_time_start

        epoch_mse_train = running_mse_train / len(features_train)
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
    
    def __evaluate_one_epoch(self,
                model: Any,
                features_test: torch.Tensor,
                targets_test: torch.Tensor
            ) -> tuple[float, float, float]:
        model.eval()
        with torch.no_grad():
            running_mse_test = 0
            running_mae_test = 0
            for _, i in enumerate(range(0, len(features_test) - 1, self.__batch_size)):
                features_test_batch = self.__get_batch(features_test, i)
                targets_test_batch = self.__get_batch(targets_test, i)

                # compute the MSE
                loss_function = torch.nn.MSELoss()
                predicted_targets_batch = model(features_test_batch)
                loss = loss_function(predicted_targets_batch, targets_test_batch)
                running_mse_test += loss.item()

                # compute the MAE
                loss_function = torch.nn.L1Loss()
                predicted_targets_batch = model(features_test_batch)
                loss = loss_function(predicted_targets_batch, targets_test_batch)
                running_mae_test += loss.item()
        
        epoch_mse_test = running_mse_test / len(features_test)
        epoch_rmse_test = np.sqrt(epoch_mse_test)
        epoch_mae_test = running_mae_test / len(features_test)
        
        return epoch_mse_test, epoch_rmse_test, epoch_mae_test
    
    def __client_fn(self, context: Context) -> Client:
        print("Running node id %s on partition id %s" % (context.node_id,context.node_config["partition-id"] ))
        partition_id = context.node_config["partition-id"]
        
        features_train = self.__features_tensor_train[partition_id]
        targets_train = self.__targets_tensor_train[partition_id]
        features_test = self.__features_tensor_test[partition_id]
        targets_test = self.__targets_tensor_test[partition_id]
    
        return FlowerClient(). \
            build_fl_model(self.__model). \
            build_fl_optimizer(self.__optimizer). \
            build_fl_features_train(features_train). \
            build_fl_targets_train(targets_train). \
            build_fl_features_test(features_test). \
            build_fl_targets_test(targets_test). \
            build_get_parameters_callback(self.__get_parameters). \
            build_set_parameters_callback(self.__set_parameters). \
            build_train_callback(self.__train_many_epochs). \
            build_test_callback(self.__evaluate_one_epoch). \
            build_predict_callback(self.__predict). \
            build(). \
            to_client()

    def __weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """Aggregate metrics using weighted MSE."""

        total_mse_test = 0.0
        total_rmse_test = 0.0
        total_mae = 0.0
        total_r2_test = 0.0
        total_examples = 0
        
        y_test_list = list()
        y_pred_list = list()
        for num_examples, metric in metrics:
            total_mse_test += num_examples * metric["mse_test"]  # MSE ponderato dal numero di esempi
            total_rmse_test += num_examples * metric["rmse_test"]  # RMSE ponderato dal numero di esempi
            total_mae += num_examples * metric["mae_test"]  # MAE ponderato dal numero di esempi
            total_r2_test += num_examples * metric["r2_test"] # R2 ponderato dal numero di esempi
            total_examples += num_examples

            y_test_json = json.loads(metric['y_test'])
            y_test = np.asarray(y_test_json["array"])
            y_test_plt = pd.DataFrame(y_test, columns=['min_cpu','max_cpu','avg_cpu'])
            y_test_list.append(y_test_plt.to_dict())

            y_pred_json = json.loads(metric['y_pred'])
            y_pred = np.asarray(y_pred_json["array"])
            y_pred_plt = pd.DataFrame(y_pred, columns=['min_cpu','max_cpu','avg_cpu'])
            y_pred_list.append(y_pred_plt.to_dict())

        self.__history['predict']['y_test'] = y_test_list
        self.__history['predict']['y_pred'] = y_pred_list
        
        aggregated_mse_test = total_mse_test / total_examples if total_examples > 0 else 0.0 # Calcola MSE medio
        aggregated_rmse_test = total_rmse_test / total_examples if total_examples > 0 else 0.0 # Calcola RMSE medio
        aggregated_mae_test = total_mae / total_examples if total_examples > 0 else 0.0 # Calcola MAE medio
        aggregated_r2_test = total_r2_test / total_examples if total_examples > 0 else 0.0 # Calcola R2 medio

        self.__history['test']['mse'].append(aggregated_mse_test)
        self.__history['test']['rmse'].append(aggregated_rmse_test)
        self.__history['test']['mae'].append(aggregated_mae_test)
        self.__history['test']['r2'].append(aggregated_r2_test)

        self.__history_fill_callback(self.__batch_size, self.__lookback, self.__history)

        return {
            "mse_test": aggregated_mse_test,
            "rmse_test": aggregated_rmse_test,
            "mae_test": aggregated_mae_test,
            "r2_test": aggregated_r2_test
        }

    def __server_fn(self, context: Context) -> ServerAppComponents:
        config = ServerConfig(num_rounds=self.__num_rounds)
        strategy = FedAvg(
            fraction_fit = self.__fraction_fit,
            fraction_evaluate = self.__fraction_evaluate,
            min_fit_clients = self.__min_fit_clients,
            min_evaluate_clients = self.__min_evaluate_clients,
            min_available_clients =self.__min_available_clients,
            evaluate_metrics_aggregation_fn=self.__weighted_average
        )
        return ServerAppComponents(strategy=strategy, config=config)

    def learn(self):
        client_app = ClientApp(client_fn=self.__client_fn)
        server_app = ServerApp(server_fn=self.__server_fn)
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
        run_simulation(
            server_app = server_app,
            client_app = client_app,
            num_supernodes = self.__num_clients,
            backend_config=backend_config
        )

    
    def __predict(self, X_test, y_test):
        with torch.no_grad():
            y_pred = self.__model(X_test)
            y_pred = self.__normalizer.inverse_transform(y_pred.cpu())
        y_test = self.__normalizer.inverse_transform(y_test)

        y_test_plt = pd.DataFrame(y_test, columns=['min_cpu','max_cpu','avg_cpu'])
        y_pred_plt = pd.DataFrame(y_pred, columns=['min_cpu','max_cpu','avg_cpu'])

        self.__history['predict']['y_test'] = y_test_plt.to_dict()
        self.__history['predict']['y_pred'] = y_pred_plt.to_dict()

        return y_test, y_pred