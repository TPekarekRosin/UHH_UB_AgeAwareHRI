import os
import time
import threading
import webrtcvad
import sys
import sounddevice as sd

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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.asr_model = WhisperModel(self.config['model_name'], device=device,
                                      compute_type="float16" if device == "cuda" else "int8")
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
        self.age_estimations = []

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
        
        microphones = list_microphones(audio)
        selected_input_device_id = get_input_device_id(device_name, microphones)
        
        """# check if the defined sample rate works with the device, and if not
        try:
            is_supported = audio.is_format_supported(sample_rate,
                                                     input_device=selected_input_device_id,
                                                     input_channels=n_channels, input_format=pa_format)
        except ValueError:
            print("Config sample rate doesn't work, setting sample rate to 16000.")
            sample_rate = 16000

        try:
            is_supported = audio.is_format_supported(sample_rate,
                                                     input_device=selected_input_device_id,
                                                     input_channels=n_channels, input_format=pa_format)
        except ValueError:
            print("16000 as sample rate doesn't work, setting sample rate to 48000.")
            sample_rate = 48000"""

        chunk_size = int(sample_rate / 10)

        print("Sample rate ", sample_rate)
        print("Chunk size ", chunk_size)
        sd.check_input_settings(selected_input_device_id, samplerate=sample_rate)

        stream = audio.open(input_device_index=selected_input_device_id,
                            format=pa_format,
                            channels=n_channels,
                            rate=sample_rate,
                            input=True,
                            frames_per_buffer=chunk_size)

        self.asr_output = False
        try:
            import speech_processing_client as spc
            spc.speech_publisher("please ready to serve", 0, 1.0)
        except rospy.ROSInterruptException:
            pass

        self.asr_output = True
        frames = b''
        speech_started = False
        while not rospy.is_shutdown():
            audio_chunk = stream.read(chunk_size, exception_on_overflow=False)
            audio_int16 = np.frombuffer(audio_chunk, np.int16)
            audio_float32 = self.int2float(audio_int16)

            new_confidence = self.vad_model(torch.from_numpy(audio_float32), 16000).item()

            # todo increase tolerance for pauses
            if new_confidence > 0.5 and self.asr_output:
                if not speech_started:
                    speech_started = True
                frames += audio_chunk
                self.confidences.append(new_confidence)
            else:
                if speech_started:
                    speech_started = False
                    print("FRAMES", len(frames))
                    if len(frames) > 1 and self.asr_output:
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
                # Publish binary age, recognized text, assumed command and confidence
                if confidence > 0.5:
                    self.age_estimations.insert(0, age_estimation.item())
                    if len(self.age_estimations) > 5:
                        self.age_estimations.pop()
                    age_estimation_mean = np.mean(self.age_estimations)
                    age = 0 if age_estimation <= 0.5 else 1
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

    def callback(self, msg):
        if msg.data == "on" and not self.asr_output:
            rospy.loginfo("ASR activated.")
            self.asr_output = True
        elif msg.data == "off" and self.asr_output:
            rospy.loginfo("ASR deactivated.")
            self.asr_output = False
        
    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / 32768
        sound = sound.squeeze()  # depends on the use case
        return sound


