import torch
import numpy as np
from typing import Any
from flwr.client import NumPyClient
from json import JSONEncoder
import json

from sklearn.metrics import r2_score

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
class FlowerClient(NumPyClient):

    __instance = None

    def __new__(self):
        if not self.__instance:
            self.__instance = super(FlowerClient, self).__new__(self)
        return self.__instance
    
    
    def build_fl_model(self, model: Any):
        self.__model = model
        return self.__instance

    def build_fl_optimizer(self, optimizer: Any):
        self.__optimizer = optimizer
        return self.__instance
    
    def build_fl_features_train(self, features: torch.Tensor):
        self.__features_train = features
        return self.__instance
    
    def build_fl_targets_train(self, targets: torch.Tensor):
        self.__targets_train = targets
        return self.__instance
    
    def build_fl_features_test(self, features: torch.Tensor):
        self.__features_test = features
        return self.__instance
    
    def build_fl_targets_test(self, targets: torch.Tensor):
        self.__targets_test = targets
        return self.__instance
    
    def build_get_parameters_callback(self, get_parameters_callback: Any):
        self.__get_parameters_callback = get_parameters_callback
        return self.__instance
    
    def build_set_parameters_callback(self, set_parameters_callback: Any):
        self.__set_parameters_callback = set_parameters_callback
        return self.__instance
    
    def build_train_callback(self, train_callback: Any):
        self.__train_callback = train_callback
        return self.__instance
    
    def build_test_callback(self, test_callback: Any):
        self.__test_callback = test_callback
        return self.__instance

    def build_predict_callback(self, predict_callback: Any):
        self.__predict_callback = predict_callback
        return self.__instance
    
    def build(self):
        return self.__instance


    def get_parameters(self, config):
        return self.__get_parameters_callback(self.__model)
    
    
    def fit(self, parameters, config):
        self.__set_parameters_callback(self.__model, parameters)
        history = self.__train_callback(
            self.__model,
            self.__optimizer,
            self.__features_train,
            self.__targets_train
        )
        return self.__get_parameters_callback(self.__model), len(self.__features_train), {}

    def evaluate(self, parameters, config):
        self.__set_parameters_callback(self.__model, parameters)
        epoch_mse_test, epoch_rmse_test, epoch_mae_test = self.__test_callback(
            self.__model,
            self.__features_test,
            self.__targets_test
        )

        y_test, y_pred = self.__predict_callback(
            self.__features_test,
            self.__targets_test
        )
        accuracy = r2_score(y_test, y_pred)
        accuracy = round(accuracy, 2) * 100
        y_test_json = {"array": y_test}
        y_test_str = json.dumps(y_test_json, cls=NumpyArrayEncoder)
        y_pred_json = {"array": y_pred}
        y_pred_str = json.dumps(y_pred_json, cls=NumpyArrayEncoder)


        return float(epoch_mse_test), \
            len(self.__features_test), \
            {
                "mse_test": float(epoch_mse_test),
                "rmse_test": float(epoch_rmse_test),
                "mae_test": float(epoch_mae_test),
                "r2_test": float(accuracy),
                "y_test": y_test_str,
                "y_pred": y_pred_str
            }