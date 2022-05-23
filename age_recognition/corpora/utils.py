import os
import soundfile as sf
import datasets
import pandas as pd
from sklearn import model_selection


# todo adapt method for general loading from huggingface
def from_huggingface(dir_path, ds_name="hf-internal-testing/librispeech_asr_dummy", val_size=0.1):
    os.makedirs(dir_path, exist_ok=True)
    ds = datasets.load_dataset(ds_name, "clean", split="validation")
    ds = ds.map(map_to_array)

    train_df = pd.DataFrame(ds)
    train, val = model_selection.train_test_split(train_df, random_state=42, test_size=val_size,
                                                  stratify=train_df['label'])
    train.to_csv(os.path.join(dir_path, 'train.csv'), header=True, index=False)
    val.to_csv(os.path.join(dir_path, 'val.csv'), header=True, index=False)

    sentences_raw = ag_news_test['text']
    labels = ag_news_test['label']
    sentences = []
    for s in list(sentences_raw):
        s = expand_contractions(s)
        s = preprocess(s)
        sentences.append(s)
    test_dct = {'sentence': list(sentences),
                'label': list(labels)}

    test_df = pd.DataFrame(test_dct)
    test_df.to_csv(os.path.join(dir_path, 'test.csv'), header=True, index=False)


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch