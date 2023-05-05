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
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import sentencepiece as spm

from .utils import get_input_device_id, \
    list_microphones, read_sentence_list, levenshtein, min_levenshtein

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

        self.sentence_list, self.tokens_list = read_sentence_list()

        self.inference_model = LiveInference(self.config)

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
        # A frame must be either 10, 20, or 30 ms in duration for webrtcvad
        frame_duration = self.config['frame_duration']
        chunk_size = int(sample_rate * frame_duration / 1000)

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
            is_speech = vad.is_speech(frame, sample_rate)
            if is_speech:
                frames += frame
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

            float64_buffer = np.frombuffer(
                audio_frames, dtype=np.int16) / 32767
            start = time.perf_counter()
            text, age_estimation = self.inference_model.buffer_to_text(float64_buffer)
            inference_time = time.perf_counter() - start
            sample_length = len(float64_buffer) / 16000  # length in sec
            if text != "":
                age = 0 if age_estimation <= 0.5 else 1
                command, confidence = self.command_recognition(text)
                # Publish binary age, recognized text, assumed command and confidence
                if confidence > 0.5:
                    try:
                        import speech_processing_client as spc
                        print("evtl circular import? ")
                        spc.speech_publisher(text, command, age, confidence)
                    except rospy.ROSInterruptException:
                        pass
                output_queue.put([text, command, confidence, age_estimation, sample_length, inference_time])

    def get_last_text(self):
        return self.asr_output_queue.get()

    def command_recognition(self, text):
        folder, _ = os.path.split(__file__)
        filepath = os.path.join(os.path.dirname(folder), 'age_recognition', 'sp_models', 'en_massive_2000.model')

        sp = spm.SentencePieceProcessor()
        sp.load(filepath)
        tokens = sp.encode_as_ids(text)

        best_match = min_levenshtein(tokens, self.tokens_list)
        distance = levenshtein(tokens, self.tokens_list[best_match])
        confidence = 1.0 - min(1.0, float(distance) / len(self.tokens_list[best_match]))

        return self.sentence_list[best_match], confidence


class LiveInference:
    def __init__(self, config):
        # todo: add functionality to load own model
        self.processor = Wav2Vec2Processor.from_pretrained(config['model_name'])
        self.model = Wav2Vec2ForCTC.from_pretrained(config['model_name'])
        
        # todo add ar_model
        # self.ar_model =

    def buffer_to_text(self, audio_buffer):
        if len(audio_buffer) == 0:
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        # todo: age estimation
        age_estimation = 0.8
        return transcription.lower(), age_estimation

    def file_to_text(self, filename):
        audio_input, samplerate = sf.read(filename)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)

