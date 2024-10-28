import flwr
import json
import torch
import logging
import itertools
import numpy as np
from os import path
import pandas as pd
from . import utils
from os import listdir
from typing import List
from .models.lstm import LSTM
from torchinfo import summary
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from .learner.SequenceLearner import SequenceLearner

class ExperimentFLLSTM():
    
    __instance = None

    def __new__(cls):
        if not cls.__instance:
            cls.__instance = super(ExperimentFLLSTM, cls).__new__(cls)
        return cls.__instance
    
    def __init__(self) -> None:
        self.__load_dataset_function_map = {
            "azure": self.__load_dataset_azure
        }
        self.__preprocessing_function_map = {
            "azure": self.__preprocessing_many_azure
        }
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Using device: %s" % (self.__device))
        logging.info(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")


    def build_configuration(self, config: dict):
        self.__dataset_name = config['dataset_name']
        self.__dataset_path = config['dataset']
        logging.info("Dataset: %s" % (self.__dataset_path))

        self.__experiment = config['experiment']

        self.__lookforward = 1
        self.__lookbacks = config['lookbacks']
        self.__num_epochs = config['num_epochs']
        self.__num_rounds = config['num_rounds']
        self.__learning_rate = config['learning_rate']
        self.__input_size = config['input_size'] # number of features
        self.__hidden_size = config['hidden_size'] # number of features in hidden state
        self.__num_layers = config['num_layers'] # number of stacked LSTM layers
        self.__num_classes = config['num_classes'] # number of output classes
        self.__batch_sizes = config['batch_size']

        return self.__instance
    
    def build_result_dir(self, result_dir: str):
        self.__result_dir = result_dir
        return self.__instance
    
    def build_metrics(self, metrics: dict):
        self.__metrics = metrics
        return self.__instance
    
    def build(self):
        logging.info("Setup the CPU Utilization Forecasting eperiments with Federated Learning LSTM ...")    
        return self.__instance
    

    def __load_dataset(self) -> pd.core.frame.DataFrame:
        return self.__load_dataset_function_map[self.__dataset_name]()
    
    def __load_dataset_azure(self):
        df_list = list()
        for filename in listdir(self.__dataset_path):
            filepath = path.join(self.__dataset_path, filename)
            if path.isfile(filepath):
                df_list.append( pd.read_csv(filepath) )
        return df_list

    def __preprocessing(self, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        return self.__preprocessing_function_map[self.__dataset_name](df)
    
    def __preprocessing_many_azure(self, df_list: List[pd.core.frame.DataFrame]) -> List[pd.core.frame.DataFrame]:
        preprocessed_df_list = list()
        for df in df_list:
            preprocessed_df_list.append(self.__preprocessing_one_azure(df))
        return preprocessed_df_list
    
    def __preprocessing_one_azure(self, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        df = df.set_index('timestamp')
        df.drop(columns=['vm_id', 'Unnamed: 0'], inplace=True)
        return df


    def learn(self):
        df_list = self.__load_dataset()
        df_list = self.__preprocessing(df_list)
        logging.info("Number of datasets: %s" % (len(df_list)))
        # logging.info("Number of samples: %s" % (len(df)))

        i = 0
        train_percentage = 0.8
        minmaxscaler = MinMaxScaler()
        train_scaled_list = list()
        test_scaled_list = list()
        for df in df_list:
            # TODO: move into a EDA
            utils.store_dataset_overview(
                title="Dataset at a glance",
                ylabel="CPU Utilization",
                data=df,
                labels=df.columns,
                filename="%s/dataset%s-overview.png" % (self.__result_dir, i)
            )
            train, test = utils.split_train_test(df, train_percentage)
            train_scaled, test_scaled = utils.normalize(minmaxscaler, train, test, df.columns)
            train_scaled_list.append(train_scaled)
            test_scaled_list.append(test_scaled)
            i += 1

        metrics = dict()

        for batch_size in self.__batch_sizes:
            logging.info("Batch size %s" % (batch_size))
            metrics[batch_size] = dict()

            for lookback in self.__lookbacks:
                logging.info("Lookback %s steps" % (lookback))

                X_train_tensor_list = list()
                y_train_tensor_list = list()
                for train_scaled in train_scaled_list:
                    X_train, y_train = utils.create_sequence(
                        train_scaled,
                        lookback=lookback,
                        lookforward=self.__lookforward
                    )
                    
                    X_train_tensors = utils.create_tensor(X_train).to(self.__device)   
                    y_train_tensors = utils.create_tensor(y_train).to(self.__device)

                    X_train_tensor_list.append(X_train_tensors)
                    y_train_tensor_list.append(y_train_tensors)
                
                X_test_tensor_list = list()
                y_test_tensor_list = list()
                for test_scaled in test_scaled_list:
                    X_test, y_test = utils.create_sequence(
                        test_scaled,
                        lookback=lookback,
                        lookforward=self.__lookforward
                    )

                    X_test_tensors = utils.create_tensor(X_test).to(self.__device)
                    y_test_tensors = utils.create_tensor(y_test).to(self.__device)

                    X_test_tensor_list.append(X_test_tensors)
                    y_test_tensor_list.append(y_test_tensors)

                model = LSTM(
                    num_classes = self.__num_classes,
                    input_size = self.__input_size,
                    hidden_size = self.__hidden_size,
                    num_layers = self.__num_layers
                ).to(self.__device)
                summary(model, input_size=(batch_size, 3, 3))

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=self.__learning_rate
                )

                learner = SequenceLearner(). \
                    build_batch_size(batch_size). \
                    build_features_train(X_train_tensors). \
                    build_targets_train(y_train_tensors). \
                    build_features_test(X_test_tensors). \
                    build_targets_test(y_test_tensors). \
                    build_num_epoch(self.__num_epochs). \
                    build_num_rounds(self.__num_rounds). \
                    build_model(model). \
                    build_optimizer(optimizer). \
                    build()
                
                learner.start()
        #         exit()
        #         history = learner.train(evaluate=True)
        #         metrics[batch_size][lookback] = history
        #         logging.info("stored metrics for %s batch size and %s lookback" % (batch_size, lookback))

        #         y_test_plt, y_pred_plt = learner.predict(minmaxscaler, y_test)
        #         metrics[batch_size][lookback]['predict']['y_test'] = y_test_plt.to_dict()
        #         metrics[batch_size][lookback]['predict']['y_pred'] = y_pred_plt.to_dict()
        #         logging.info("stored predictions for %s batch size and %s lookback" % (batch_size, lookback))

        # data_path = self.__result_dir + "data.json"
        # logging.info("Storing metrics on %s" % (data_path))
        # with open(data_path, 'w') as f:
        #     json.dump(metrics, f)


    def __metrics_figure(self, result_dir, batch_size, dataset, metric) -> None:
        marker = itertools.cycle(('o', 'v', '^', '<', '>', 's')) 
        plt.figure(figsize=(16, 8))
        plt.title("Model: %s | Batch size: %s | Dataset: %s" % (
                self.__experiment.upper(),
                batch_size,
                self.__dataset_name
            ),
            fontsize=30
        )
        plt.xlabel("epochs", fontsize=24)
        plt.xticks(fontsize=20)
        plt.ylabel(metric.upper(), fontsize=24)
        plt.yticks(fontsize=20)
        for lookback in self.__lookbacks:
            lookback = str(lookback)
            plt.plot(
                self.__metrics[batch_size][lookback][dataset][metric],
                marker = next(marker),
                label="%s steps" % (lookback)
            )
        plt.legend(fontsize="20")
        plt.savefig("%s/%s-%s-%s-%s.png" % (result_dir, self.__dataset_name, batch_size, metric, dataset))
        plt.close()
        return
    
    def __prediction_figure(self, result_dir, batch_size, lookback, y_test_plt, y_pred_plt, label) -> None:
        plt.figure(figsize=(16, 8))
        plt.title(
            "Model: %s | Batch size: %s | Lookback: %s | Dataset: %s" % (
                self.__experiment.upper(),
                batch_size,
                lookback,
                self.__dataset_name
            ),
            fontsize=30
        )
        plt.xlabel("steps", fontsize=24)
        plt.xticks(fontsize=20)
        plt.ylabel(label, fontsize=24)
        plt.yticks(fontsize=20)
        plt.plot(
            np.asarray(y_test_plt.index, float),
            y_test_plt[label],
            label="actual",
            marker = 'o'
        )
        plt.plot(
            np.asarray(y_pred_plt.index, float),
            y_pred_plt[label],
            label="predicted",
            marker = 's'
        )
        plt.legend(fontsize="20")
        plt.savefig('%s/%s-%s-%sbatch-%ssteps' % (result_dir, self.__dataset_name, label, batch_size, lookback))
        plt.close()

    def plot(self, result_dir) -> None:
        logging.info("Store metrics figures.")
        for batch_size in self.__batch_sizes:
            batch_size = str(batch_size)

            self.__metrics_figure(result_dir, batch_size, 'train', 'mse')
            self.__metrics_figure(result_dir, batch_size, 'train', 'rmse')
            self.__metrics_figure(result_dir, batch_size, 'test', 'mse')
            self.__metrics_figure(result_dir, batch_size, 'test', 'rmse')
            self.__metrics_figure(result_dir, batch_size, 'test', 'mae')

            for lookback in self.__lookbacks:
                lookback = str(lookback)
                y_test_plt = pd.DataFrame.from_dict(self.__metrics[batch_size][lookback]['predict']['y_test'])
                y_pred_plt = pd.DataFrame.from_dict(self.__metrics[batch_size][lookback]['predict']['y_pred'])

                self.__prediction_figure(result_dir, batch_size, lookback, y_test_plt, y_pred_plt, 'max_cpu')
                self.__prediction_figure(result_dir, batch_size, lookback, y_test_plt, y_pred_plt, 'min_cpu')
                self.__prediction_figure(result_dir, batch_size, lookback, y_test_plt, y_pred_plt, 'avg_cpu')


    def __create_row(self, batch_size, dataset, metric) -> None:
        row = {
                'model': self.__experiment,
                'metric': metric
            }
        for lookback in self.__lookbacks:
            lookback = str(lookback)
            row[lookback] = self.__metrics[batch_size][lookback][dataset][metric][-1]
        return row

    def table(self, result_dir) -> None:
        logging.info("Store metrics table.")
        
        for batch_size in self.__batch_sizes:
            batch_size = str(batch_size)

            train_table = pd.DataFrame(columns=['model', 'metric', '2', '6', '12', '24', '48', '96'])
            test_table = pd.DataFrame(columns=['model', 'metric', '2', '6', '12', '24', '48', '96'])

            row = self.__create_row(batch_size, 'train', 'mse')
            train_table = pd.concat([train_table, pd.DataFrame([row])], ignore_index=True)
            row = self.__create_row(batch_size, 'train', 'rmse')
            train_table = pd.concat([train_table, pd.DataFrame([row])], ignore_index=True)

            row = self.__create_row(batch_size, 'test', 'mse')
            test_table = pd.concat([test_table, pd.DataFrame([row])], ignore_index=True)
            row = self.__create_row(batch_size, 'test', 'rmse')
            test_table = pd.concat([test_table, pd.DataFrame([row])], ignore_index=True)
            row = self.__create_row(batch_size, 'test', 'mae')
            test_table = pd.concat([test_table, pd.DataFrame([row])], ignore_index=True)

            train_table_path = result_dir + '%s-%s-train_table' % (self.__dataset_name, batch_size)
            with open(train_table_path, 'w') as f:
                f.write(train_table.to_latex(index=False))

            test_table_path = result_dir + '%s-%s-test_table' % (self.__dataset_name, batch_size)
            with open(test_table_path, 'w') as f:
                f.write(test_table.to_latex(index=False))