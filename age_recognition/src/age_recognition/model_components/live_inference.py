import os
import yaml
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

folder, _ = os.path.split(__file__)
filepath = os.path.join(os.path.dirname(folder), 'configs', 'live_model_config.yaml')
with open(filepath) as f:
    config = yaml.safe_load(f)


class LiveInference:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(config['model_name'])
        # todo: add functionality to load own model
        self.model = Wav2Vec2ForCTC.from_pretrained(config['model_name'])

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
