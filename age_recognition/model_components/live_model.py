import os
import time
import threading
import webrtcvad

import numpy as np
import pyaudio as pa
import torch
import rospy
from queue import Queue
from dp.phonemizer import Phonemizer

from age_recognition.model_components.live_inference import LiveInference
from age_recognition.model_components.utils import get_input_device_id, \
    list_microphones, read_sentence_list, levenshtein, min_levenshtein
from speech_processing.src.speech_client import speech_recognized_client, age_recognition_publisher

# TODO: add hotword detection


class ASRLiveModel:
    exit_event = threading.Event()

    def __init__(self, device_name="default", **config):
        self.device_name = device_name
        self.config = config

        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()

        self.asr_process = threading.Thread(target=self.asr_process,
                                            args=(self.asr_input_queue,
                                                  self.asr_output_queue))
        self.vad_process = threading.Thread(target=self.vad_process,
                                            args=(self.device_name,
                                                  self.asr_input_queue,))

        self.phonemizer = Phonemizer.from_checkpoint('checkpoints/best_model.pt')
        self.sentence_list, self.phoneme_list = read_sentence_list(self.phonemizer)

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
        while True:
            if ASRLiveModel.exit_event.is_set():
                break
            frame = stream.read(chunk_size)
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
        while True:
            audio_frames = in_queue.get()
            if audio_frames == "close":
                break

            float64_buffer = np.frombuffer(
                audio_frames, dtype=np.int16) / 32767
            start = time.perf_counter()
            # ROS call
            text, age_estimation = speech_recognized_client(float64_buffer)
            inference_time = time.perf_counter() - start
            sample_length = len(float64_buffer) / 16000  # length in sec
            if text != "":
                age = 0 if age_estimation <= 0.5 else 1
                # todo: only return command for confidence values above threshold value
                command, confidence = self.command_recognition(text)
                # Publish binary age and recognized command
                try:
                    age_recognition_publisher(command, age)
                except rospy.ROSInterruptException:
                    pass

                output_queue.put([text, command, confidence, age_estimation, sample_length, inference_time])

    def get_last_text(self):
        # returns the text, sample length and inference time in seconds.
        return self.asr_output_queue.get()

    def command_recognition(self, text):
        ph_text = self.phonemizer(text, lang='en_us')

        best_match = min_levenshtein(ph_text, self.phoneme_list)
        distance = levenshtein(ph_text, self.phoneme_list[best_match])
        confidence = 1.0 - min(1.0, float(distance) / len(self.phoneme_list[best_match]))

        return self.sentence_list[best_match], confidence
