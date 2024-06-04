import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision.transforms import Normalize
import torchaudio.transforms as T
from typing import Optional, Tuple, Union
import logging
from transformers import WhisperFeatureExtractor, WhisperPreTrainedModel, WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperDecoderLayer

from .ar_components import ModifiedXVector
from .utils import SequenceClassifierOutput
from .attention_network import MHAClassifier, MultiHeadAttentionLayer

logger = logging.getLogger('Age-Estimator-Log')
_HIDDEN_STATES_START_POSITION = 1


class AgeEstimation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        whisper_config = WhisperConfig.from_pretrained("openai/whisper-small",
                                                       num_labels=self.config['model']['num_classes'])
        whisper_config.num_classification_layers = self.config['num_classification_layers']
        whisper_config.num_classifier_attention_heads = self.config['num_classifier_attention_heads']

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained()
        self.classification_model = ClassificationWhisperModel.from_pretrained(self.config['ar_model'],
                                                                              config=whisper_config)

    def forward(self, waveform):
        input_features = self.feature_extractor(waveform, sampling_rate=16000)
        outputs = self.classification_model(input_features)
        return outputs.logits


class ClassificationWhisperModel(WhisperPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = WhisperEncoder(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        self.projector = MHAClassifier(MultiHeadAttentionLayer(config.hidden_size,
                                                               num_heads=config.num_classifier_attention_heads),
                                       nn.Linear(config.hidden_size, config.classifier_proj_size), config)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # save config separately for forward funtion
        self.classification_config = config

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training. Only the projection layers and classification head will be updated.
        """
        self.encoder._freeze_parameters()

    def get_input_embeddings(self) -> nn.Module:
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.encoder.set_input_embeddings(value)

    def forward(
            self,
            input_features: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if self.config.use_weighted_layer_sum:
            output_hidden_states = True
        elif output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if self.config.use_weighted_layer_sum:
            hidden_states = encoder_outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = encoder_outputs[0]

        hidden_states = self.projector(hidden_states)
        pooled_output = hidden_states.mean(dim=1)

        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
