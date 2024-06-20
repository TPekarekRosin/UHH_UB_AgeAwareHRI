import os
import yaml
import logging
from tqdm import tqdm

import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.utils import data
from faster_whisper import WhisperModel
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import evaluate

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

    test_dataloader = data.DataLoader(test['test'], batch_size=1, shuffle=False)

    # create asr model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    asr_model = WhisperModel(config['model_name'], device=device,
                                      compute_type="float16" if device == "cuda" else "int8")
    # create ar model
    ar_model = AgeEstimation(config)

    # load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # todo create metric logging: wer, cer, accuracy, confusion matrix, inference time

    age_accuracies, age_true, age_pred, wers, cers = [], [], [], [], []
    for it, batch in tqdm(enumerate(test_dataloader)):
        audio_path = batch['audio']['path'][0]
        audio_array = batch['audio']['array']
        sentence = batch['sentence'][0]
        age_label = batch['label'].item()
        segments, _ = asr_model.transcribe(audio_path)
        s = list(segments)
        text = s[0].text
        ar_out = ar_model(audio_array.numpy())
        age_estimation = torch.argmax(ar_out, dim=-1).item()
        if age_estimation == age_label:
            age_accuracies.append(1)
        else:
            age_accuracies.append(0)

        wers.append(100 * wer_metric.compute(predictions=[text], references=[sentence]))
        cers.append(100 * cer_metric.compute(predictions=[text], references=[sentence]))

        age_true.append(age_label)
        age_pred.append(age_estimation)
        # print(text, sentence)
        # print(age_estimation, age_label)

    print("ACCURACY: {0}, WER: {1}, CER: {2}".format(np.mean(age_accuracies),
                                                     np.mean(wers),
                                                     np.mean(cers)))
    cm = confusion_matrix(y_true=age_true, y_pred=age_pred)
    print("CONFUSION MATRIX ", cm)


if __name__ == '__main__':
    logger.info('Starting Evaluation of all modules...')
    # asr and ar:
    logger.info('... Automatic Speech Recognition and Age Recognition')
    logger.info('-----------------------------------------------------')
    eval_asr_ar()
