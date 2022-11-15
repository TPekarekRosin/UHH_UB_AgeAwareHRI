#!/usr/bin/env python3
import os
import soundfile as sf
import editdistance
import numpy as np
import sentencepiece as spm


def get_input_device_id(device_name, microphones):
    for device in microphones:
        if device_name in device[1]:
            return device[0]


def list_microphones(pyaudio_instance):
    info = pyaudio_instance.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    result = []
    for i in range(0, numdevices):
        if (pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            name = pyaudio_instance.get_device_info_by_host_api_device_index(
                0, i).get('name')
            result += [[i, name]]
    return result


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


def read_sentence_list():
    folder, _ = os.path.split(__file__)
    filepath = os.path.join(os.path.dirname(folder), 'age_recognition', 'configs', 'commands.txt')

    sp = spm.SentencePieceProcessor()
    sp_filepath = os.path.join(os.path.dirname(folder), 'age_recognition', 'sp_models', 'en_massive_2000.model')
    sp.load(sp_filepath)

    sentence_list, tokens_list = [], []
    with open(filepath, "r") as f:
        for sentence in f:
            sentence = sentence.rstrip('\n')
            sentence_list.append(sentence.lower())
            tokens = sp.encode_as_ids(sentence)
            tokens_list.append(tokens)
    return sentence_list, tokens_list


def levenshtein(a, b):
    # Calculates the Levenshtein distance between a and b
    return editdistance.eval(a, b)


def min_levenshtein(input_seq, list_of_seqs):
    best_matching = np.argmin([levenshtein(input_seq, comp_seq) for comp_seq in list_of_seqs])
    return best_matching



