import logging
from flwr.client import NumPyClient

class FlowerClient(NumPyClient):
    def __init__(self, model, features_train, features_test, get_parameters_callback, set_parameters_callback, train_callback, test_callback):
        self.__model = model
        self.__features_train = features_train
        self.__features_test = features_test
        self.__get_parameters_callback = get_parameters_callback
        self.__set_parameters_callback = set_parameters_callback
        self.__train_callback = train_callback
        self.__test_callback = test_callback

    def get_parameters(self, config):
        return self.__get_parameters_callback(self.__model)

    def fit(self, parameters, config):
        logging.info("Sono su FIT")

        self.__set_parameters_callback(self.__model, parameters)
        self.__train_callback()
        return self.__get_parameters_callback(self.__model), len(self.__features_train), {}

    def evaluate(self, parameters, config):
        self.__set_parameters_callback(self.__model, parameters)
        epoch_mse_test, epoch_rmse_test, epoch_mae_test = self.__test_callback()
        return float(epoch_mse_test), len(self.__features_test), {"accuracy": float(1)}