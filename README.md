# CPU Utilization Forecasting with Federated Learning
![python-3.10.16](https://img.shields.io/badge/python-3.10.16-blue)

This project studies the effectiveness of resolving the prediction of computational resources by using the Federated Learning. Specifically, a Bi-LSTM in Federated Learning is trained over the CPU utilization of Microsoft Azure virtual machines. The results show a promising quality of the model in terms of generalization.

## Install Dependencies
The only packages you need in your operating system are Python3 and PIP3. The experiment was run on `Python v3.10.16`, `pip v25.0.1` and `virtualenv==20.29.2`. For the installation of these dependencies, please refer to the official documentation.

## Getting Started
Install all Python3 dependencies with the `requirements.txt` file as follow:
```bash
python3.10 -m virtualenv -p python3.10 venv
source venv/bin/activate
pip install -r requirements.txt
```

Execute the application as follow:
```bash
python main.py --config conf.yaml
```

The `Flower` simulator will run a number of clients and a server with the parameters defined in the configuration file called `conf.yaml`. During the execution, a folder called `results` is made. The folder contains the overview of each and every dataset, that is the plot of minimum, average and maximum CPU utilization for the selected VMs. Moreover, the folder contains the file `data.json` with the numeric results of the experiment. Indeed, for each batch and for each input window size, test and predict numeric values are collected. The test includes results for the MSE, RMSE, MAE and R-squared metrics. The predict include the actual values and the predicted values of the minimum, average and maximum CPU utilization.

In order to make figures out of the `data.json` file, run the following within the virtual environment:
```bash
python main.py --config conf.yaml --metrics results
```

As result, for each batch, figures about the metrics validated in the test dataset are created, along with a LaTeX table with the latest values for each metric. Moreover, for each batch, for each input window size and for each dataset, a figure that compares the actual values and the predicted values of the minimum, average and maximum CPU utilization are created.

## Datasets
The datasets are extracted from the [AzurePublicDataset](https://github.com/Azure/AzurePublicDataset) repository, a public collection of Microsoft Azure traces for the benefit of the research and academic community. Specifically, this repository uses the [AzurePublicDatasetV2](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md), which contains a representative subset of the first-party Azure Virtual Machine (VM) workload in one geographical region. Considering the dataset is very large with more than 2.5 million of VMs involved, we extracted a sample of 7 VMs.

After the running of the applications, the `results` folder will include PNG files called `datasetX-overview.png`, where X is in the closed range [0, 5].

## Configuration file
Use the YAML configuration file to finetune the experiment. What is not considered as a parameter in the configuration file is static (e.g., the Federated Learning aggregation strategy). The configuration file `conf.yaml` is described as follow:
```yaml
experiment: <string> # the name of the experiment
dataset_name: <string> # the name of the dataset
dataset: <string> # the relative path of the datasets folder
lookbacks: <list<int>> # the list of input window size
num_epochs: <int> # the number of epochs
num_rounds: <int> # the number of rounds
learning_rate: <float> # the learning rate
input_size: <int> # the input size according to the LSTM model
hidden_size: <int> # the hidden size of the LSTM model
num_layers: <int> # the number of layers of the LSTM model
num_classes: <int> # the number of classes of the LSTM model
batch_size: <list<int>> # the list of batch size in which the dataset should be splitted
num_clients: <int> # the number of clients
fraction_fit: <int> # the sample N% of available clients for training
fraction_evaluate: <int> # the sample N% of available clients for evaluation
min_fit_clients: <int> # never sample less than N clients for training
min_evaluate_clients: <int> # never sample less than N clients for evaluation
min_available_clients: <int> # wait until all N clients are available
```

## Metrics
The followin are the metrics studied and validated in the experiment:
- the $R^2$ measures the proportion of the variance in the dependent variable that is predictable from the independent variables, providing an indication of the model's accuracy and is defined as $$R^2 = 1 - \frac{{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}}{{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$ where $y_i$ represents the real value, $\bar{y}$ represents the average of the real values, and $\hat{y}$ the predicted value. The calculated value $R^2$ must lie between 0 and 1: a value close to 1 indicates a higher predictive capacity. A negative value $R^2$ implies that the model is not usable.
- the Mean Squared Error (MSE) measures the average of squared errors between predicted values and actual values. It emphasizes larger errors more than MAE and is commonly used for regression tasks. It is defined as follow: $ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2$
- the Root Mean Squared Error (RMSE) measures the standard deviation between predicted values and actual values. It is useful for understanding the absolute error when the errors are squared to prevent positive and negative values from canceling each other out. It is defined as follow: $\text{RMSE} = \sqrt{MSE}$
- the Mean Absolute Error (MAE) measures the average of the absolute errors between predicted values and actual values. It is useful for understanding the accuracy of a model's predictions: $ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|$

After running the experiment, the `data.json` file is created in the `results` folder. Running again the `main.py` application with the `--metrics` option will go through the JSON file to make figures of the metrics. They will be called: `azure-256-mae-test.png`, `azure-256-mse-test.png`, `azure-256-rmse-test.png` and `azure-256-r2-test.png`.

## Credits
This project is the result of a joint research collaboration between three Insitute:
- University of Messina, Italy
- MT Atlantique, Nantes Université, Ecole Centrale Nantes, CNRS, Inria, LS2N, France
- University of Utah, United States of America

As product of the research, a paper was accepted in the <i>25th IEEE International Symposium on Cluster, Cloud, and Internet Computing (CCGrid 2025)</i> titled <i>Private Distributed Resource Management Data: Predicting CPU Utilization with Bi-LSTM and Federated Learning</i> by Lorenzo Carnevale, Daniel Balouek, Serena Sebbio, Manish Parashar and Massimo Villari.