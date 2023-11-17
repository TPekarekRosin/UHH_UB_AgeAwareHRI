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
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import sentencepiece as spm

from .utils import get_input_device_id, \
    list_microphones, levenshtein, min_levenshtein
from .age_recognition_model import AgeEstimation

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class ASRLiveModel:
    exit_event = threading.Event()
    
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
        
        # age recognition
        self.ar_model = AgeEstimation(self.config)
        filepath = os.path.join(os.path.dirname(folder), 'src', 'age_recognition', 'configs',
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
        time.sleep(5)
        # start the voice activity detection
        self.vad_process.start()
    
    def stop(self):
        # stop asr
        ASRLiveModel.exit_event.set()
        self.asr_input_queue.put("close")
        print("asr stopped")
    
    # voice activity detector
    def vad_process(self, device_name, asr_input_queue):
        vad = webrtcvad.Vad()
        vad.set_mode(1)
        
        audio = pa.PyAudio()
        pa_format = pa.paInt16
        n_channels = self.config['n_channels']
        sample_rate = self.config['sample_rate']
        chunk_size = 1024
        
        microphones = list_microphones(audio)
        selected_input_device_id = get_input_device_id(device_name, microphones)
        
        stream = audio.open(input_device_index=selected_input_device_id,
                            format=pa_format,
                            channels=n_channels,
                            rate=sample_rate,
                            input=True,
                            frames_per_buffer=chunk_size)
        
        frames = b''
        while not rospy.is_shutdown():
            if ASRLiveModel.exit_event.is_set():
                break
            frame = stream.read(chunk_size, exception_on_overflow=False)
            float32_buffer = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            new_confidence = self.vad_model(torch.from_numpy(float32_buffer), 16000).item()
            
            if new_confidence > 0.6:
                frames += frame
                self.confidences.append(new_confidence)
            else:
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
            if audio_frames == "close":
                break
            
            float32_buffer = np.frombuffer(audio_frames, dtype=np.int16).astype(np.float32)
            # speech recognition
            segments, _ = self.asr_model.transcribe(float32_buffer)
            s = list(segments)
            
            if len(s) != 0:
                text = s[0].text
                confidence = np.mean(self.confidences)
                # age recognition
                ar_out = self.ar_model(torch.from_numpy(float32_buffer))
                age_estimation = torch.argmax(ar_out, dim=-1) / 100.0
                age = 0 if age_estimation <= 0.5 else 1
                # Publish binary age, recognized text, assumed command and confidence
                if confidence > 0.5:
                    try:
                        import speech_processing_client as spc
                        spc.speech_publisher(text, age, confidence)
                    except rospy.ROSInterruptException:
                        pass
                output_queue.put([text, confidence, age_estimation])
                self.confidences = []
    
    def get_last_text(self):
        return self.asr_output_queue.get()



