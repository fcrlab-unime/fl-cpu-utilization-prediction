__copyright__ = 'Copyright 2024, FCRLab at University of Messina'
__author__ = 'Lorenzo Carnevale <lcarnevale@unime.it>'
__maintainers__ = 'Serena Sebbio <serena.sebbio@unime.it>'
__credits__ = 'Unversity of Messina (Italy)'
__description__ = 'CPU utilization prediction by using LSTM Federated Learning model'

import os
import json
import yaml
import logging
import argparse
from experiments.experiment_lstm import ExperimentFLLSTM

def setup_logging(logdir_name: str) -> None:
    format = "%(asctime)s %(filename)s:%(lineno)d %(levelname)s - %(message)s"
    datefmt = "%d/%m/%Y %H:%M:%S"
    level = logging.INFO
    filename = '%s/experiments.log' % logdir_name
    logging.basicConfig(filename=filename, format=format, level=level, datefmt=datefmt)

def main():
    description = ('%s\n%s\n%s' % (__author__, __maintainers__, __description__))
    epilog = ('%s\n%s' % (__credits__, __copyright__))
    parser = argparse.ArgumentParser(
        description = description,
        epilog = epilog
    )

    logdir_name = 'logs'
    if not os.path.exists(logdir_name):
        os.makedirs(logdir_name)
    setup_logging(logdir_name)

    parser.add_argument('-c', '--config',
                        dest='config',
                        help='YAML configuration file',
                        type=str,
                        required=True)
    
    parser.add_argument('-m', '--metrics',
                        dest='metrics',
                        help='Metrics JSON file',
                        type=str)
    
    options = parser.parse_args()

    experiments = {
        'fllstm': ExperimentFLLSTM,
    }

    with open(options.config) as f:
        config = yaml.safe_load(f)

    # results_dir = 'results-%s-%s/' % (config['experiment'], config['dataset_name'])
    results_dir = 'results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if options.metrics:
        with open(options.metrics) as f:
            metrics = json.load(f)
        experiment = experiments[config['experiment']](). \
            build_configuration(config). \
            build_metrics(metrics). \
            build()
        experiment.plot(results_dir) # produce plots
        experiment.table(results_dir) # produce table
        return
    
    if options.config:
        experiment = experiments[config['experiment']](). \
            build_configuration(config). \
            build_result_dir(results_dir). \
            build()
        experiment.learn() # produce metrics JSON file

    return

if __name__ == "__main__":
    main()