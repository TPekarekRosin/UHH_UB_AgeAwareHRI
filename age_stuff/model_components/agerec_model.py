import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, HubertModel
from datasets import load_dataset
from utils import map_to_array


class AgeRecModel(nn.Module):
    def __init__(self, n_classes, pooling, device):
        super(AgeRecModel, self).__init__()
        self.device = device

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.transformer = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        # linear output layer for classification into 4 age groups
        if n_classes is not None:
            self.linear = nn.Linear(self.transformer.config.hidden_size, n_classes)

        assert pooling in ['mean', 'cls', 'max', 'first-last-avg']
        self.transformer.config.pooling = pooling
        self.pooling = pooling

        self.to(self.device)

    def forward(self, inputs, out_from='full'):
        # todo: add and test different embedding pooling
        # Entire pipeline: process -> encoder -> linear ->logits
        if out_from == 'full':
            input_values = self.processor(inputs, return_tensors="pt").input_values
            hidden_states = self.transformer(input_values).last_hidden_state
            out = self.linear(hidden_states)
        # Speech encoding only: process -> encoder -> embedding
        elif out_from == 'transformer':
            input_values = self.processor(inputs, return_tensors="pt").input_values
            hidden_states = self.transformer(input_values).last_hidden_state
            out = hidden_states
        elif out_from == 'linear':
            out = self.linear(inputs)
        else:
            raise NotImplementedError

        return out


if __name__ == '__main__':
    # test with dummy data
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.map(map_to_array)

    device = torch.device('cpu')
    ar_model = AgeRecModel(4, 'mean', device)

    logits = ar_model(ds["speech"][0], out_from='full')
    predicted_class_ids = torch.argmax(logits, dim=-1)

    print(predicted_class_ids)
