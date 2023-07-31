import torch
from torch import nn
from torchvision.transforms import Normalize
import torchaudio.transforms as T
import logging

from .ar_components import ModifiedXVector

logger = logging.getLogger('Age-Estimator-Log')


class AgeEstimation(nn.Module):
    def __init__(self, config):
        super(AgeEstimation, self).__init__()
        self.speaker_embeds = SpeakerEmbeddingExtractor(config)
        self.estimator = AgeEstimationClassification(config)

    def forward(self, waveform):
        embedding = self.speaker_embeds(waveform)
        out = self.estimator(embedding)
        return out


class SpeakerEmbeddingExtractor(nn.Module):
    def __init__(self, config):
        super(SpeakerEmbeddingExtractor, self).__init__()
        self.feature_extraction = T.MFCC(sample_rate=config['model']['sample_rate'],
                                         n_mfcc=config['model']['n_mfcc'],
                                         melkwargs={'win_length': config['model']['win_length'],
                                                    'hop_length': config['model']['hop_length']})
        # self.batch_norm = nn.BatchNorm1d(num_features=config['model']['n_mfcc'])
        self.x_vector = ModifiedXVector(input_dim=config['model']['n_mfcc'])
        self.fc1 = nn.Linear(1024, config['model']['fc_hidden_dim'])
        self.config = config
        self.device = torch.device(
            "cuda:" + str(self.config['training']['gpu']) if torch.cuda.is_available() else "cpu")
    
        # for lmcl phase 2 :
        self.nllloss = nn.CrossEntropyLoss()
        self.fc2 = nn.Linear(config['model']['fc_hidden_dim'], config['model']['fc_output_dim'])

    def forward(self, waveform):
        mfcc = self.feature_extraction(waveform)  # [0])
        norm = Normalize(mean=[0 for _ in range(len(waveform))], std=[1 for _ in range(len(waveform))])
        mfcc_norm = norm(mfcc)
        mfcc_norm_t = mfcc_norm.transpose(1, 2)
    
        pooled_out = self.x_vector(mfcc_norm_t)
        fc1_out = self.fc1(pooled_out)
        out = self.fc2(fc1_out)
        
        return out


class AgeEstimationClassification(nn.Module):
    def __init__(self, config):
        super(AgeEstimationClassification, self).__init__()
        self.linear1 = nn.Linear(config['model']['fc_output_dim'],
                                 config['model']['fc_output_dim'])
        self.linear2 = nn.Linear(config['model']['fc_output_dim'],
                                 config['model']['fc_output_dim'] // 2)
        self.linear3 = nn.Linear(config['model']['fc_output_dim'] // 2,
                                 config['model']['num_classes'])

        self.activation = nn.LeakyReLU()

        # returns class log probabilities
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        linear1_out = self.linear1(x)
        linear2_out = self.linear2(linear1_out)
        linear3_out = self.linear3(linear2_out)
        activated_out = self.activation(linear3_out)
        log_probs = self.log_softmax(activated_out)
        
        return log_probs