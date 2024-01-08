import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch
import utils 
from transformers import AutoFeatureExtractor
import utils

feature_extractor = AutoFeatureExtractor.from_pretrained(utils.MODEL_PATH)

def collate_fn(batch):
    # Séparer les inputs et les labels
    inputs, labels, language_ids = zip(*batch)

    # Padding des inputs pour qu'ils aient tous la même longueur
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0.0)

    # Convertir les labels et language_ids en tenseurs
    labels = torch.tensor(labels, dtype=torch.long)
    language_ids = torch.tensor(language_ids, dtype=torch.long)

    return inputs_padded, labels, language_ids

def replace_labels(example):
    label_mapping = {0: 2, 5: 3, 4: 0, 3: 1}
    
    example['label'] = label_mapping.get(example['label'], example['label'])
    return example

def is_label_to_keep(example):
    return example['label'] not in [1, 2, 6]

def pack_audio_data(row):
    return {
        'array': np.array(row['audio']),           # les données audio
        'sampling_rate': row['sampling_rate']  # le taux d'échantillonnage
    }

def normalize_emodb(X, y):
    data = {'audio': X[0],'sampling_rate': X[1], 'label': y}
    df = pd.DataFrame(data)

    valeurs_a_supprimer = ['A', 'E', 'L']

    masque = ~df['label'].isin(valeurs_a_supprimer)

    df_EMODB = df[masque]
    df_EMODB['label'] = df_EMODB['label'].replace({'W': 2, 'T': 3, 'N': 0, 'F': 1})
    return df_EMODB

def normalize_mesd(X, y):
    data = {'audio': X[0],'sampling_rate': X[1] ,'label': y}
    df = pd.DataFrame(data)

    valeurs_a_supprimer = ['Disgust', 'Fear']

    masque = ~df['label'].isin(valeurs_a_supprimer)

    df_MESD = df[masque]
    df_MESD['label'] = df_MESD['label'].replace({'Anger': 2, 'Sadness': 3, 'Neutral': 0, 'Happiness': 1})
    return df_MESD

def normalize_emouerj(X, y):
    data = {'audio': X[0], 'sampling_rate': X[1], 'label': y}
    df_EMOUERJ = pd.DataFrame(data)
    df_EMOUERJ['label'] = df_EMOUERJ['label'].replace({'a': 2, 's': 3, 'n': 0, 'h': 1})
    return df_EMOUERJ

def split_dataset(dataset):
    train_test_split = dataset.train_test_split(test_size=0.3)
    test_dataset = train_test_split['test']
    test_val_split = test_dataset.train_test_split(test_size=0.5)
    return train_test_split['train'], test_val_split['train'], test_val_split['test']
    
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = utils.feature_extractor(
        audio_arrays, sampling_rate=utils.feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    return inputs
