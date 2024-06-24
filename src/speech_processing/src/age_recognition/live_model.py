import os
import time
import threading
import webrtcvad
import sys
import sounddevice as sd
import signal
from scipy.signal import resample

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
            
        self.evaluation_mode = self.config['eval_mode']
        
        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()

        # Register the signal handler for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self.signal_handler)
        
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
        self.sample_rate = 16000
        
        # age recognition
        self.ar_model = AgeEstimation(self.config)

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
        
        microphones = list_microphones(audio)
        selected_input_device_id = get_input_device_id(device_name, microphones)
        dev_info = audio.get_device_info_by_index(selected_input_device_id)
        print(f"Device {selected_input_device_id}: {dev_info['name']}")
        print(f"\t- Input channels: {dev_info['maxInputChannels']}")
        print(f"\t- Supported sampling rates: {dev_info['defaultSampleRate']}")

        # n_channels = dev_info['maxInputChannels']
        n_channels = 1
        self.sample_rate = int(dev_info['defaultSampleRate'])

        chunk_size = 2048 if self.sample_rate == 16000 else 4096     # round(sample_rate / 16000) * 2048    # int(sample_rate / 10)

        print("Sample rate ", self.sample_rate)
        print("Chunk size ", chunk_size)

        stream = audio.open(input_device_index=selected_input_device_id,
                            format=pa_format,
                            channels=n_channels,
                            rate=self.sample_rate,
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
        pause_buffer = 0
        speech_started = False
        print('\nSpeak!\n')
        while not rospy.is_shutdown():
            audio_chunk = stream.read(chunk_size, exception_on_overflow=False)
            audio_float32 = self.convert_to_float(audio_chunk)
            audio_float32_resampled = self.resample_audio(audio_float32, self.sample_rate, 16000)

            new_confidence = self.vad_model(torch.from_numpy(audio_float32_resampled), 16000).item()
            if new_confidence > 0.5 and self.asr_output:
                if not speech_started:
                    speech_started = True
                frames += audio_chunk
                self.confidences.append(new_confidence)
            else:
                # print(new_confidence)
                if speech_started and pause_buffer >= 5:
                    speech_started = False
                    # print("FRAMES", len(frames))
                    if self.asr_output:  # and len(frames) > 1
                        asr_input_queue.put(frames)
                elif speech_started and pause_buffer < 5:
                    pause_buffer += 1
                    continue
                elif not speech_started:
                    frames = b''
                    pause_buffer = 0

        stream.stop_stream()
        stream.close()
        audio.terminate()
    
    def asr_process(self, in_queue, output_queue):
        while not rospy.is_shutdown():
            audio_frames = in_queue.get()
            audio_float32 = self.convert_to_float(audio_frames)
            audio_float32_resampled = self.resample_audio(audio_float32, self.sample_rate, 16000)

            # speech recognition
            segments, _ = self.asr_model.transcribe(audio_float32_resampled)
            s = list(segments)
            
            if len(s) != 0:
                text = s[0].text
                confidence = np.mean(self.confidences)
                # age recognition
                ar_out = self.ar_model(torch.from_numpy(audio_float32))
                age_estimation = torch.argmax(ar_out, dim=-1) / 9.0
                # Publish binary age, recognized text, assumed command and confidence
                if confidence > 0.5:
                    self.age_estimations.insert(0, age_estimation.item())
                    if len(self.age_estimations) > 5:
                        self.age_estimations.pop()
                    age_estimation_mean = np.mean(self.age_estimations)
                    age = 0 if age_estimation_mean <= 0.5 else 1
                    if self.evaluation_mode:
                        self.log_results(age, age_estimation, text)
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

    def convert_to_float(self, audio_chunk):
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0

    def signal_handler(self, sig, frame):
        # Perform cleanup actions here, if needed
        print("Ctrl+C detected. Exiting gracefully.")
        # Optionally, you can raise SystemExit to exit the program
        raise SystemExit

    def resample_audio(self, audio_data, original_rate, target_rate):
        # Resample audio data to the target sampling rate in real-time
        ratio = target_rate / original_rate
        num_samples = int(len(audio_data) * ratio)
        resampled_audio = resample(audio_data, num_samples)
        return resampled_audio

    def resample_audio_linear(self, audio_data, original_rate, target_rate):
        # Calculate resampling ratio
        ratio = target_rate / original_rate

        # Determine the number of samples for resampling
        num_samples = int(len(audio_data) * ratio)

        # Generate new indices for resampling
        indices = np.arange(num_samples) / ratio

        # Perform linear interpolation
        resampled_audio = np.interp(indices, np.arange(len(audio_data)), audio_data).astype(np.float32)

        return resampled_audio
    
    def log_results(self, age, age_estimation, text):
        folder, _ = os.path.split(__file__)
        path = os.path.dirname(folder)
        
        file_path = ""
        if not os.path.exists(file_path):
            os.make
            