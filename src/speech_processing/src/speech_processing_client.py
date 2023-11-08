#!/usr/bin/env python3
import os
import sys
import yaml
import torch
import rospy
import pyaudio as pa
import numpy as np
from speech_processing.msg import *
# from age_recognition.live_model import ASRLiveModel
from faster_whisper import WhisperModel
# from speech_processing.msg import command_and_age
from age_recognition.utils import get_input_device_id, \
    list_microphones, levenshtein, min_levenshtein
from age_recognition.age_recognition_model import AgeEstimation

"""sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))"""


def speech_publisher(transcript, age, confidence):
    pub = rospy.Publisher('speech_publisher', command_and_age, queue_size=10)
    # rospy.init_node('custom_age_talker', anonymous=True)
    r = rospy.Rate(10)  # 10hz

    msg = command_and_age()
    msg.transcript = transcript
    msg.age = age
    msg.confidence = confidence

    #while not rospy.is_shutdown():
    rospy.loginfo(msg)
    pub.publish(msg)
    r.sleep()


class ASRLiveModel:
    def __init__(self, pyaudio_instance, device_name="default"):
        self.p = pyaudio_instance
        self.device_name = device_name

        folder, _ = os.path.split(__file__)
        cfg_filepath = os.path.join(os.path.dirname(folder), 'src', 'age_recognition', 'configs', 'live_model_config.yaml')
        # cfg_filepath = './age_recognition/configs/live_model_config.yaml'
        with open(cfg_filepath) as f:
            self.config = yaml.safe_load(f)

        # voice activity detection
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=True)
        # speech recognition
        self.asr_model = WhisperModel(self.config['model_name'], device="cuda", compute_type="float16")

        # age recognition
        self.ar_model = AgeEstimation(self.config)
        filepath = os.path.join(os.path.dirname(folder), 'src', 'age_recognition', 'configs', self.config['ar_checkpoint'])
        pretrained_dict = torch.load(filepath)
        model_dict = self.ar_model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.ar_model.load_state_dict(pretrained_dict)

        self.last_asr_output = ['', 0.0, 0]

    def live_inference(self):
        pa_format = pa.paInt16
        n_channels = self.config['n_channels']
        sample_rate = self.config['sample_rate']
        chunk_size = 1024

        microphones = list_microphones(self.p)
        selected_input_device_id = get_input_device_id(self.device_name, microphones)

        print("Recording")

        stream = self.p.open(input_device_index=selected_input_device_id,
                        format=pa_format,
                        channels=n_channels,
                        rate=sample_rate,
                        frames_per_buffer=chunk_size,
                        input=True)

        speech_started = False
        frames = b''
        confidences = []
        while not rospy.is_shutdown():
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames += data

            float32_buffer = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            new_confidence = self.vad_model(torch.from_numpy(float32_buffer), 16000).item()

            # if no speech was detected...
            if new_confidence < 0.6:
                print("NC", new_confidence)
                # ... but previously speech was detected, we add all segments and
                # pass it to the dialogue manager
                if speech_started:
                    segments, _ = self.asr_model.transcribe(float32_buffer)
                    s = list(segments)
                    # make sure no false positives slip through
                    if len(s) != 0:
                        print(s)
                        text = s[0].text
                        confidence = np.mean(confidences)
                        ar_out = self.ar_model(torch.from_numpy(float32_buffer))
                        age_estimation = torch.argmax(ar_out, dim=-1) / 100.0
                        age = 0 if age_estimation <= 0.5 else 1
                        try:
                            print("Publishing", text, confidence, age)
                            speech_publisher(text, age, confidence)
                        except rospy.ROSInterruptException:
                            print("Passing")
                            pass
                        # update queue
                        self.last_asr_output = [text, confidence, age_estimation]
                    speech_started = False
                frames = b''
                confidences = []
            else:
                if not speech_started:
                    speech_started = True
                    confidences.append(new_confidence)

    def get_last_text(self):
        return self.last_asr_output


if __name__ == "__main__":
    rospy.init_node('speech_engine', anonymous=True)

    # choose audio device
    p = pa.PyAudio()
    print('Available audio input devices:')
    input_devices = []
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels'):
            input_devices.append(i)
            print(i, dev.get('name'))

    if len(input_devices):
        dev_idx = -2
        while dev_idx not in input_devices:
            print('Please type input device ID:')
            dev_idx = int(input())
    device = p.get_device_info_by_index(dev_idx)
    asr = ASRLiveModel(p, device.get('name'))

    while not rospy.is_shutdown():
        asr.live_inference()
