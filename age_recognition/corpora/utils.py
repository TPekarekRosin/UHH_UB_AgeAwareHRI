import os
import soundfile as sf
import datasets
import wandb
import pandas as pd
from sklearn import model_selection
from age_vox_celeb import AgeVoxCelebDataset
#from common_voice import CommonVoiceDataset

import plotly.express as px
import plotly.graph_objects as go


def load_dataset(base_path, name):
    if name.upper() == 'CV':
        dir_path = os.path.join(base_path, 'data/common_voice')
        if not os.path.exists(os.path.join(dir_path, 'common_voice')):
            from_huggingface(dir_path, ds_name='')
        train = AmazonSlsDataset(os.path.join(dir_path, 'amazon/train.csv'))
        val = AmazonSlsDataset(os.path.join(dir_path, 'amazon/val.csv'))
        test = AmazonSlsDataset(os.path.join(dir_path, 'amazon/test.csv'))
    elif name.upper() == 'AGE_VOX_CELEB':
        dir_path = os.path.join(base_path, 'data/uci_sentiment')
        if not os.path.exists(os.path.join(dir_path, 'imdb')):
            download_uci_sentiment(dir_path)
        train = ImdbSlsDataset(os.path.join(dir_path, 'imdb/train.csv'))
        val = ImdbSlsDataset(os.path.join(dir_path, 'imdb/val.csv'))
        test = ImdbSlsDataset(os.path.join(dir_path, 'imdb/test.csv'))
    else:
        raise Exception('The requested dataset name {} is not valid. Check typos or implementation.'.format(name))

    return train, val, test


# todo adapt method for general loading from huggingface
def from_huggingface(dir_path, ds_name="hf-internal-testing/librispeech_asr_dummy", val_size=0.1):
    os.makedirs(dir_path, exist_ok=True)
    ds = datasets.load_dataset(ds_name, "clean", split="train")
    ds = ds.map(map_to_array)

    train_df = pd.DataFrame(ds)
    train, val = model_selection.train_test_split(train_df, random_state=42, test_size=val_size,
                                                  stratify=train_df['label'])
    train.to_csv(os.path.join(dir_path, 'train.csv'), header=True, index=False)
    val.to_csv(os.path.join(dir_path, 'val.csv'), header=True, index=False)

    ds_test = datasets.load_dataset(ds_name, "clean", split="test")
    ds_test = ds_test.map(map_to_array)

    test_df = pd.DataFrame(ds_test)
    test_df.to_csv(os.path.join(dir_path, 'test.csv'), header=True, index=False)

    return train, val, test_df


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


def plot_cv(path):
    df = pd.read_csv(path)
    print(df[:5])
    fig_age = px.histogram(df, x="age", category_orders=dict(
        age=["twenties", "thirties", "forties", "fifties", "sixties", "seventies", "eighties", "nineties", ""]
    ))

    wandb.log({'CV-Age': fig_age})

    fig_gender = px.histogram(df, x="age", category_orders=dict(
        gender=["female", "male", ""]
    ))

    wandb.log({'CV-Gender': fig_gender})
