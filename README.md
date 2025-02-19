# CPU Utilization Forecasting with Federated Learning
![python-3.9.6](https://img.shields.io/badge/python-3.9.6-blue)

This project studies the effectiveness of resolving the resorce managemengent (e.g., CPU) problem using the Federated Learning. 

The application was tested on two machines:
- MacBook Pro with chip Apple M3 Pro, 18GB of memory RAM and Sonoma 14.6.1
- ...

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

The `Flower` simulator will run a number of clients and a server with the parameters defined in the configuration file called `conf.yaml`.

## Datasets
The datasets are extracted from the [AzurePublicDataset](https://github.com/Azure/AzurePublicDataset) repository, a public collection of Microsoft Azure traces for the benefit of the research and academic community. Specifically, this repository uses the [AzurePublicDatasetV2](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md), which contains a representative subset of the first-party Azure Virtual Machine (VM) workload in one geographical region. Considering the dataset is very large with more than 2.5 million of VMs involved, we extracted a sample of 7 VMs.

## Credits
This project is the result of a joint research collaboration between three Insitute:
- University of Messina, Italy
- MT Atlantique, Nantes Université, Ecole Centrale Nantes, CNRS, Inria, LS2N, France
- University of Utah, United States of America

The following is a list of pubblication related with this repository:
- ...