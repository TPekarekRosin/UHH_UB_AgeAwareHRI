import os
import yaml
import logging

import torch
from faster_whisper import WhisperModel
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from src.speech_processing.src.age_recognition.age_recognition_model import AgeEstimation
from eval_tools.eval_utils import load_test_dataset_hf, preprocess

logger = logging.getLogger('Evaluation Logger')


def eval_asr_ar():
    # load configs
    folder, _ = os.path.split(__file__)
    filepath = os.path.join(os.path.dirname(folder), 'src', 'speech_processing', 'src', 'age_recognition',
                            'configs', 'live_model_config.yaml')
    eval_filepath = os.path.join(os.path.dirname(folder), 'eval_tools', 'eval_config.yaml')
    with open(filepath) as f:
        config = yaml.safe_load(f)
    with open(eval_filepath) as f:
        eval_config = yaml.safe_load(f)
    config.update(eval_config)
    print(config)

    # load dataset for eval
    test = load_test_dataset_hf(config, keep_in_memory=False)
    feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-small')
    tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-small', task="transcribe")

    def _prepare_ds_input(batch):
        new_batch = {}
        new_batch["label"] = batch["label"]

        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]
        sentence = [preprocess(batch['sentence'][x]) for x in range(len(batch['sentence']))]

        # compute log-Mel input features from input audio array
        new_batch["input_features"] = \
            [feature_extractor(audio[x]["array"], sampling_rate=audio[x]["sampling_rate"]).input_features[0]
             for x in range(len(audio))]

        # encode target text to label ids
        new_batch["labels"] = tokenizer(sentence, padding=True).input_ids
        return new_batch
    test = test.with_transform(_prepare_ds_input)

    # create asr model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    asr_model = WhisperModel(config['model_name'], device=device,
                                      compute_type="float16" if device == "cuda" else "int8")
    # create ar model
    ar_model = AgeEstimation(config)

    # todo create metric logging: wer, cer, accuracy, confusion matrix, inference time

    # todo create inference loop over dataset


if __name__ == '__main__':
    logger.info('Starting Evaluation of all modules...')
    # asr and ar:
    logger.info('... Automatic Speech Recognition and Age Recognition')
    logger.info('-----------------------------------------------------')
    eval_asr_ar()
