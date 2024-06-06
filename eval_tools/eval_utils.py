import os
import re
import logging
import random
import yaml
from pydub import AudioSegment
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from datasets import load_dataset, DatasetDict, Audio, interleave_datasets, Dataset, concatenate_datasets
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

from corpora.audio_dataset import AudioDataset
from corpora.text_dataset import KeywordDataset

logger = logging.getLogger('Eval-Utils-Log')


def load_test_dataset_hf(config, keep_in_memory=True):
    logger.info('Loading CV-Age Dataset.')
    path = config['dataset']['folder_path']
    train_meta = config['dataset']['train_ds']
    audio_folder = config['dataset']['audio_folder']
    dataset = DatasetDict()

    split = load_dataset("csv", data_files=os.path.join(path, train_meta), split='train',
                         delimiter="\t" if train_meta.split('.')[-1] == 'tsv' else ",")

    ds_splits = split.train_test_split(test_size=0.3, seed=config['training']['seed'])
    dataset['test'] = ds_splits['test']

    def change_to_server_path(batch):
        # if the meta file already contains a local file path for each file change it to the path on the server,
        # if not simply attach the server path
        _path = path
        local_path = batch['path']

        if '/' in local_path:
            file_id = local_path.split('/')[-1]
        else:
            file_id = local_path

        batch['path'] = os.path.join(_path, audio_folder, file_id)

        return batch

    dataset['test'] = dataset['test'].map(lambda batch: change_to_server_path(batch))

    column_names = dataset['test'].column_names
    if 'audio' not in column_names:
        dataset['test'] = dataset['test'].cast_column("path", Audio(sampling_rate=16000))
        dataset['test'] = dataset['test'].rename_column('path', 'audio')

    removable_column_names = [x for x in column_names if x not in
                              ['audio', 'sentence', config['model']['classification']]]

    dataset['test'] = dataset['test'].remove_columns(removable_column_names)
    if config['dataset']['classification'] != "label":
        dataset['test'] = dataset['test'].rename_column(config['dataset']['classification'], "label")
        print(dataset)

    config['dataset']['keep_in_memory'] = keep_in_memory

    return dataset


def preprocess(text):
    # lower-case characters
    text = text.lower()
    # add space after $
    text = text.replace(r'$', r'$ ')
    # remove explicit \n
    text = re.sub(r'(\\n)+', r' ', text)
    # replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)
    # remove &=<NUMBER> (possibly emoticons)
    text = re.sub(r'&#[0-9]+', '', text)
    # remove @name references and hyperlinks
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', r'', text)
    # isolate and remove punctuations except '?'
    text = re.sub(r'([$\"\.\(\)\!\?\\\/\,])', r' \1 ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    # specifically for sst-2: remove -LRB- and -RRB-
    text = re.sub(r'-LRB-|-RRB-', ' ', text).strip()
    # remove multiple and trailing whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
