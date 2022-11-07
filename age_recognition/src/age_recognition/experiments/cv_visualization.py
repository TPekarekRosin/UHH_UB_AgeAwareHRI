import os
import logging

import torchaudio.datasets
import yaml
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
import random
import numpy as np
from importlib import import_module

from corpora.utils import plot_cv

# Init command line logging
logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt="%m/%d/%y %H:%M:%S")
logger = logging.getLogger('CV-Visualization-Log')


folder, _ = os.path.split(__file__)
filepath = os.path.join(os.path.dirname(folder), 'configs', 'cv_visualization.yaml')
with open(filepath) as f:
    config = yaml.safe_load(f)

# Init Weights&Biases logging
try:
    import wandb
    config['online_logging'] = True
    online_logging = True
    wandb.login(key='3f0251b768655574695cc2f6838d75d1d001096c')
    logger.info('Weights and Biases enabled.')
except:
    config['online_logging'] = False
    logger.info('Weights and Biases initialization failed. Please update environment variables accordingly.')

logger.info('Using configuration: {}'.format(config))

# Set random state
torch.manual_seed(config['seed'])
random.seed(config['seed'])
np.random.seed(config['seed'])

# Set base path
base_path = os.path.dirname(folder)
logger.info('Working directory: {}'.format(base_path))

# Set processing device
if torch.cuda.is_available():
    device = torch.device('cuda:' + str(config['gpu']))
    logger.info('There are {} GPU(s) available.'.format(torch.cuda.device_count()))
    logger.info('Using GPU ID {}: {}'.format(config['gpu'], torch.cuda.get_device_name()))
else:
    device = torch.device('cpu')
    logger.info('No GPU available, using the CPU instead.')

if __name__ == '__main__':

    logger.info('Visualization of the English Common Voice Dataset age distribution.')
    dir_path = '/informatik3/wtm/datasets/External Datasets/speech_and_language/common_voice_de/' \
               'cv-corpus-7.0-2021-07-21-de/de/'

    # Load dataset
    # cv_test = torchaudio.datasets.COMMONVOICE(dir_path, 'test.tsv')

    if config['online_logging']:
        wandb.init(project="uhh_ub_collab", entity="tpr", group="Frozen-Transformer-Experiment")
        plot_cv(dir_path + "test.tsv")
        wandb.finish()







