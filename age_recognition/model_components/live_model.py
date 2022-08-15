import time
import threading
import webrtcvad

import numpy as np
import pyaudio as pa
import torch
from queue import Queue
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf


class ASRLiveModel:
    exit_event = threading.Event()

    def __init__(self, model_name, device_name="default"):
        self.device_name = device_name
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        # todo: add call for own trained model
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

    def buffer_to_text(self, audio_buffer):
        if len(audio_buffer) == 0:
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.lower()

    def file_to_text(self, filename):
        audio_input, samplerate = sf.read(filename)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)

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

    def asr_process(self, in_queue, output_queue):

        print("\nAb jetzt kann gesprochen werden!\n")
        while True:
            audio_frames = in_queue.get()
            if audio_frames == "close":
                break

            float64_buffer = np.frombuffer(
                audio_frames, dtype=np.int16) / 32767
            start = time.perf_counter()
            text = self.buffer_to_text(float64_buffer).lower()
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

    def adjust_for_ambient_noise(self, source, duration=1):
        """
        Adjusts the energy threshold dynamically using audio from ``source`` (an ``AudioSource`` instance) to account for ambient noise.
        Intended to calibrate the energy threshold with the ambient energy level. Should be used on periods of audio without speech - will stop early if any speech is detected.
        The ``duration`` parameter is the maximum number of seconds that it will dynamically adjust the threshold for before returning. This value should be at least 0.5 in order to get a representative sample of the ambient noise.
        """
        assert isinstance(source, AudioSource), "Source must be an audio source"
        assert source.stream is not None, "Audio source must be entered before adjusting, see documentation for ``AudioSource``; are you using ``source`` outside of a ``with`` statement?"
        assert self.pause_threshold >= self.non_speaking_duration >= 0

        seconds_per_buffer = (source.CHUNK + 0.0) / source.SAMPLE_RATE
        elapsed_time = 0

        # adjust energy threshold until a phrase starts
        while True:
            elapsed_time += seconds_per_buffer
            if elapsed_time > duration: break
            buffer = source.stream.read(source.CHUNK)
            energy = audioop.rms(buffer, source.SAMPLE_WIDTH)  # energy of the audio signal

            # dynamically adjust the energy threshold using asymmetric weighted average
            damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer  # account for different chunk sizes and rates
            target_energy = energy * self.dynamic_energy_ratio
            self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)
