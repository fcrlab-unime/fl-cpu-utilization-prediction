# CPU Utilization Forecasting with Federated Learning
![python-3.9.6](https://img.shields.io/badge/python-3.9.6-blue)

This project studies the effectiveness of resolving the resorce managemengent (e.g., CPU) problem using the Federated Learning. 

The application was tested on two machines:
- MacBook Pro with chip Apple M3 Pro, 18GB of memory RAM and Sonoma 14.6.1
- ...

## Install Dependencies
The only package you need in your operating system are Python3 and PIP3. The experiment was run on `Python v3.9.6` and `virtualenv==20.26.2`.

## Getting Started
Install all Python3 dependencies with the `requirements.txt` file as follow:
```bash
python3 -m virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Execute the application as follow:
```bash
python main.py --config conf.yaml
```

The `Flower` simulator will run a number of clients and a server with the parameters defined in the configuration file called `conf.yaml`. During the execution, a folder called `results-EXPERIMENT_NAME-DATASET_NAME` is made, according to the `experiment` and `dataset_name` variables defined in the configuration file. The folder contains the overview of each and every dataset, that is the plot of minimum, average and maximum CPU utilization for the selected VMs. Moreover, the folder contains the file `data.json` with the numeric results of the experiment. Indeed, for each batch and for each input window size, test and predict numeric values are collected. The test includes results for the MSE, RMSE, MAE and R-squared metrics. The predict include the actual values and the predicted values of the minimum, average and maximum CPU utilization.

In order to make figures out of the `data.json` file, run the following with the virtual environment:
```bash
python main.py --config conf.yaml --metrics results-EXPERIMENT_NAME-DATASET_NAME
```

As result, for each batch, figures about the metrics validated in the test dataset are created, along with a LaTeX table with the latest values for each metric. Moreover, for each batch, for each input window size and for each dataset, a figure that compares the actual values and the predicted values of the minimum, average and maximum CPU utilization are created.


## Datasets
The datasets are extracted from the [AzurePublicDataset](https://github.com/Azure/AzurePublicDataset) repository, a public collection of Microsoft Azure traces for the benefit of the research and academic community. Specifically, this repository uses the [AzurePublicDatasetV2](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md), which contains a representative subset of the first-party Azure Virtual Machine (VM) workload in one geographical region. Considering the dataset is very large with more than 2.5 million of VMs involved, we extracted a sample of 7 VMs.

## Credits
This project is the result of a joint research collaboration between three Insitute:
- University of Messina, Italy
- MT Atlantique, Nantes Université, Ecole Centrale Nantes, CNRS, Inria, LS2N, France
- University of Utah, United States of America

The following is a list of pubblication related with this repository:
- ...