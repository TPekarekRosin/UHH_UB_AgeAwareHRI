import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, HubertForCTC
from datasets import load_dataset


class ASRModel(nn.Module):
    def __init__(self, sampling_rate, device):
        super(ASRModel, self).__init__()
        self.device = device

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.transformer = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

        self.sampling_rate = sampling_rate

        self.to(self.device)

    def forward(self, inputs):
        inputs = self.processor(inputs, sampling_rate=self.sampling_rate, return_tensors="pt")
        out = self.transformer(**inputs).logits
        return out


if __name__ == '__main__':
    # transcription test
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate

    device = torch.device('cpu')

    asr_model = ASRModel(sampling_rate, device)

    logits = asr_model(dataset[0]["audio"]["array"])
    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = asr_model.processor.batch_decode(predicted_ids)
    print("Transcription: ", transcription[0])
