import os
import time
import threading
import webrtcvad
import sys

import numpy as np
import pyaudio as pa
import rospy
import yaml
from queue import Queue
from faster_whisper import WhisperModel
import torch

from .utils import get_input_device_id, \
    list_microphones, levenshtein, min_levenshtein
from .age_recognition_model import AgeEstimation

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class ASRLiveModel:
    
    def __init__(self, device_name="default"):
        self.device_name = device_name
        
        folder, _ = os.path.split(__file__)
        filepath = os.path.join(os.path.dirname(folder), 'age_recognition', 'configs', 'live_model_config.yaml')
        with open(filepath) as f:
            self.config = yaml.safe_load(f)
        
        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()
        
        self.asr_process = threading.Thread(target=self.asr_process,
                                            args=(self.asr_input_queue,
                                                  self.asr_output_queue))
        self.vad_process = threading.Thread(target=self.vad_process,
                                            args=(self.device_name,
                                                  self.asr_input_queue,))

        # voice activity detection
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                               model='silero_vad',
                                               force_reload=True)
        # speech recognition
        self.asr_model = WhisperModel(self.config['model_name'], device="cuda", compute_type="float16")
        self.asr_output = True
        
        # age recognition
        self.ar_model = AgeEstimation(self.config)
        filepath = os.path.join(os.path.dirname(folder), 'age_recognition', 'configs',
                                self.config['ar_checkpoint'])
        pretrained_dict = torch.load(filepath)
        model_dict = self.ar_model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.ar_model.load_state_dict(pretrained_dict)

        self.confidences = []

    def start(self):
        # start the asr process
        self.asr_process.start()
        # start the voice activity detection
        self.vad_process.start()
    
    # voice activity detector
    def vad_process(self, device_name, asr_input_queue):
        audio = pa.PyAudio()
        pa_format = pa.paInt16
        n_channels = self.config['n_channels']
        sample_rate = self.config['sample_rate']
        chunk_size = int(sample_rate/10)
        
        microphones = list_microphones(audio)
        selected_input_device_id = get_input_device_id(device_name, microphones)
        
        stream = audio.open(input_device_index=selected_input_device_id,
                            format=pa_format,
                            channels=n_channels,
                            rate=sample_rate,
                            input=True,
                            frames_per_buffer=chunk_size)
        
        frames = b''
        speech_started = False
        while not rospy.is_shutdown():
            audio_chunk = stream.read(chunk_size, exception_on_overflow=False)
            audio_int16 = np.frombuffer(audio_chunk, np.int16)
            audio_float32 = self.int2float(audio_int16)

            new_confidence = self.vad_model(torch.from_numpy(audio_float32), 16000).item()
            
            if new_confidence > 0.5:
                if not speech_started:
                    speech_started = True
                frames += audio_chunk
                self.confidences.append(new_confidence)
            else:
                if speech_started:
                    speech_started = False
                    if len(frames) > 1:
                        asr_input_queue.put(frames)
                frames = b''
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
    def asr_process(self, in_queue, output_queue):
        print("\nSpeak!\n")
        while not rospy.is_shutdown():
            audio_frames = in_queue.get()

            audio_int16 = np.frombuffer(audio_frames, np.int16)
            audio_float32 = self.int2float(audio_int16)
            # speech recognition
            segments, _ = self.asr_model.transcribe(audio_float32)
            s = list(segments)
            
            if len(s) != 0:
                text = s[0].text
                confidence = np.mean(self.confidences)
                # age recognition
                ar_out = self.ar_model(torch.from_numpy(audio_float32))
                age_estimation = torch.argmax(ar_out, dim=-1) / 100.0
                age = 0 if age_estimation <= 0.5 else 1
                # Publish binary age, recognized text, assumed command and confidence
                if confidence > 0.5:
                    try:
                        if self.asr_output:
                            import speech_processing_client as spc
                            spc.speech_publisher(text, age, confidence)
                        else:
                            print("ASR output is disabled.")
                    except rospy.ROSInterruptException:
                        pass
                output_queue.put([text, confidence, age_estimation])
                self.confidences = []
    
    def get_last_text(self):
        return self.asr_output_queue.get()
    
    def deactivate_asr(self):
        self.asr_output = False
        
    def activate_asr(self):
        self.asr_output = True

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / 32768
        sound = sound.squeeze()  # depends on the use case
        return sound


