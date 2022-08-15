import time
import threading
import webrtcvad

import numpy as np
import pyaudio as pa
import torch
from queue import Queue
from age_recognition.model_components.live_inference import LiveInference
from speech_processing.src.speech_client import speech_recognized_client


class ASRLiveModel:
    exit_event = threading.Event()

    def __init__(self, model_name, device_name="default"):
        self.model_name = model_name
        self.device_name = device_name

    def start(self):
        """start the asr process"""
        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()
        self.asr_process = threading.Thread(target=self.asr_process,
                                            args=(self.model_name,
                                                  self.asr_input_queue,
                                                  self.asr_output_queue))
        self.asr_process.start()
        time.sleep(5)  # start vad after asr model is loaded
        self.vad_process = threading.Thread(target=self.vad_process, args=(
            self.device_name, self.asr_input_queue,))
        self.vad_process.start()

    def stop(self):
        """stop the asr process"""
        ASRLiveModel.exit_event.set()
        self.asr_input_queue.put("close")
        print("asr stopped")

    def vad_process(self, device_name, asr_input_queue):
        vad = webrtcvad.Vad()
        vad.set_mode(1)

        audio = pa.PyAudio()
        FORMAT = pa.paInt16
        CHANNELS = 1
        RATE = 16000
        # A frame must be either 10, 20, or 30 ms in duration for webrtcvad
        FRAME_DURATION = 30
        CHUNK = int(RATE * FRAME_DURATION / 1000)
        RECORD_SECONDS = 50

        microphones = self.list_microphones(audio)
        selected_input_device_id = self.get_input_device_id(
            device_name, microphones)

        stream = audio.open(input_device_index=selected_input_device_id,
                            format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        frames = b''
        while True:
            if ASRLiveModel.exit_event.is_set():
                break
            frame = stream.read(CHUNK)
            is_speech = vad.is_speech(frame, RATE)
            if is_speech:
                frames += frame
            else:
                if len(frames) > 1:
                    asr_input_queue.put(frames)
                frames = b''
        stream.stop_stream()
        stream.close()
        audio.terminate()

    def asr_process(self, model_name, in_queue, output_queue):
        wave2vec_asr = LiveInference(model_name)

        print("\nAb jetzt kann gesprochen werden!\n")
        while True:
            audio_frames = in_queue.get()
            if audio_frames == "close":
                break

            float64_buffer = np.frombuffer(
                audio_frames, dtype=np.int16) / 32767
            start = time.perf_counter()
            text = wave2vec_asr.buffer_to_text(float64_buffer).lower()
            inference_time = time.perf_counter() - start
            sample_length = len(float64_buffer) / 16000  # length in sec
            if text != "":
                output_queue.put([text, sample_length, inference_time])

    def get_input_device_id(self, device_name, microphones):
        for device in microphones:
            if device_name in device[1]:
                return device[0]

    def list_microphones(self, pyaudio_instance):
        info = pyaudio_instance.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        result = []
        for i in range(0, numdevices):
            if (pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = pyaudio_instance.get_device_info_by_host_api_device_index(
                    0, i).get('name')
                result += [[i, name]]
        return result

    def get_last_text(self):
        """returns the text, sample length and inference time in seconds."""
        return self.asr_output_queue.get()
