import librosa
import os
import pandas as pd
from datasets import load_dataset, Audio, concatenate_datasets, Dataset, Value
import preprocess_data as pp
from datasets import DatasetDict
import random

file_path_emodb = 'Data/EmoDB/'
file_path_emouerj = 'Data/emoUERJ/'
file_path_mesd = 'Data/MESD/'
file_path_oreau = 'Data/OreauFR_02/'
file_path_emovo = 'Data/EMOVO/'

def load_iemocap():
    dataset_iemocap = load_dataset("minoosh/IEMOCAP_Speech_dataset")
    all_sessions = [dataset_iemocap[session] for session in dataset_iemocap.keys()]
    ds_iemocap = concatenate_datasets(all_sessions)
    ds_iemocap = ds_iemocap.remove_columns("TURN_NAME")
    ds_iemocap = ds_iemocap.cast_column("audio", Audio(sampling_rate=16_000))
    ds_iemocap = ds_iemocap.rename_column("emotion", "label")
    ds_iemocap = ds_iemocap.cast_column('label', Value(dtype='int64'))
    ds_iemocap = ds_iemocap.add_column("language_id", [0] * len(ds_iemocap))
    return ds_iemocap


def load_subesco():
    dataset_subesco = load_dataset("sajid73/SUBESCO-audio-dataset", split="train")
    columns_to_remove = ['file name','transcription', 'speaker_id', 'speaker_name', 'speaker_gender', 'sentence_no', 'repetation_no']
    ds_subesco = dataset_subesco.remove_columns(columns_to_remove)
    ds_subesco = ds_subesco.filter(pp.is_label_to_keep)
    ds_subesco = ds_subesco.cast_column("audio", Audio(sampling_rate=16_000))
    ds_subesco = ds_subesco.map(pp.replace_labels)
    ds_subesco = ds_subesco.cast_column('label', Value(dtype='int64'))
    ds_subesco = ds_subesco.add_column("language_id", [1] * len(ds_subesco))
    return ds_subesco

def load_dataset_files(dataset_name, path):
    audios = []
    sampling_rates=[]
    labels = []

    for filename in os.listdir(path):
        if filename.endswith(".wav"):
            filepath = os.path.join(path, filename)
            audio, sampling_rate = librosa.load(filepath, sr=None)
           
            if dataset_name.upper() =='EMODB':
                emotion = filename[5]
            elif dataset_name.upper() =='MESD':
                emotion = filename.split("_")[0]
            elif dataset_name.upper() =='EMOUERJ':
                emotion = filename[3]
            else: 
                raise Exception("Nom du dataset non specifie")

            audios.append(audio)
            sampling_rates.append(sampling_rate)
            labels.append(emotion)

    X, y = [audios,sampling_rates], labels

    if dataset_name.upper() =='EMODB':
        df = pp.normalize_emodb(X, y)
        df['audio'] = df.apply(pp.pack_audio_data, axis=1)
        ds = Dataset.from_pandas(df)
        ds = ds.remove_columns(['__index_level_0__', 'sampling_rate'])
        ds = ds.cast_column('audio', Audio(sampling_rate=16_000))
        ds = ds.cast_column('label', Value(dtype='int64'))
        ds = ds.add_column("language_id", [2] * len(ds)) 

    elif dataset_name.upper() =='MESD':
        df = pp.normalize_mesd(X, y)
        df['audio'] = df.apply(pp.pack_audio_data, axis=1)
        ds = Dataset.from_pandas(df)
        ds = ds.remove_columns(['__index_level_0__', 'sampling_rate'])
        ds = ds.cast_column('audio', Audio(sampling_rate=16_000))
        ds = ds.cast_column('label', Value(dtype='int64'))
        ds = ds.add_column("language_id", [3] * len(ds)) 

    elif dataset_name.upper() =='EMOUERJ':
        df = pp.normalize_emouerj(X, y)
        df['audio'] = df.apply(pp.pack_audio_data, axis=1)
        ds = Dataset.from_pandas(df)
        ds = ds.remove_columns(['sampling_rate'])
        ds = ds.cast_column('audio', Audio(sampling_rate=16_000))
        ds = ds.cast_column('label', Value(dtype='int64'))
        ds = ds.add_column("language_id", [4] * len(ds)) 
    else: 
        raise Exception("Nom du dataset non specifie") 

    return ds

def load_emovo_dataset(path):
    audios = []
    labels = []
    sampling_rates=[]

    for subdir in os.listdir(path):
        subpath = os.path.join(path, subdir)

        if os.path.isdir(subpath):
            for filename in os.listdir(subpath):
                if filename.endswith(".wav"):
                    filepath = os.path.join(subpath, filename)
                    audio, sampling_rate = librosa.load(filepath, sr=None)
                    emotion = filename.split("-")[0]
                    audios.append(audio)
                    sampling_rates.append(sampling_rate)
                    labels.append(emotion)

    data = {'audio': [audios,sampling_rates][0],'sampling_rate': [audios,sampling_rates][1],'label': labels}
    df = pd.DataFrame(data)
    valeurs_a_supprimer = ['sor', 'dis', 'pau']
    masque = ~df['label'].isin(valeurs_a_supprimer)

    df_EMOVO = df[masque]
    df_EMOVO['label'] = df_EMOVO['label'].replace({'rab': 2, 'tri': 3, 'neu': 0, 'gio': 1})
    df_EMOVO['audio'] = df_EMOVO.apply(pp.pack_audio_data, axis=1)

    ds_emovo = Dataset.from_pandas(df_EMOVO)
    ds_emovo = ds_emovo.remove_columns(['__index_level_0__', 'sampling_rate'])
    ds_emovo = ds_emovo.cast_column('audio', Audio(sampling_rate=16_000))
    ds_emovo = ds_emovo.cast_column('label', Value(dtype='int64'))
    ds_emovo = ds_emovo.add_column("language_id", [5] * len(ds_emovo)) 
    return ds_emovo

def load_oreau_dataset(path):
    audios = []
    labels = []
    sampling_rates=[]

    for subdir in os.listdir(path):
        subpath = os.path.join(path, subdir)
        if os.path.isdir(subpath):
            for subsubdir in os.listdir(subpath):
                subsubpath = os.path.join(subpath, subsubdir)
                if os.path.isdir(subsubpath):
                    for filename in os.listdir(subsubpath):
                        if filename.endswith(".wav"):
                            filepath = os.path.join(subsubpath, filename)
                            audio, sampling_rate = librosa.load(filepath, sr=None)
                            emotion = filename[5:7]
                            audios.append(audio)
                            sampling_rates.append(sampling_rate)
                            labels.append(emotion)

    data = {'audio': [audios,sampling_rates][0],'sampling_rate': [audios,sampling_rates][1], 'label': labels}
    df = pd.DataFrame(data)

    valeurs_a_supprimer = ['Pa', 'Da', 'Sa']

    masque = ~df['label'].isin(valeurs_a_supprimer)

    df_OREAU = df[masque]
    df_OREAU['label'] = df_OREAU['label'].replace({'Ca': 2, 'Ta': 3, 'Na': 0, 'Ja': 1})
    df_OREAU['audio'] = df_OREAU.apply(pp.pack_audio_data, axis=1)

    ds_oreau = Dataset.from_pandas(df_OREAU)
    ds_oreau = ds_oreau.remove_columns(['__index_level_0__', 'sampling_rate'])
    ds_oreau = ds_oreau.cast_column('audio', Audio(sampling_rate=16_000))
    ds_oreau = ds_oreau.cast_column('label', Value(dtype='int64'))
    ds_oreau = ds_oreau.add_column("language_id", [6] * len(ds_oreau)) 
    return ds_oreau


def load_superset():
    ds_iemocap = load_iemocap()
    ds_subesco = load_subesco()
    ds_emodb = load_dataset_files('emodb', file_path_emodb)
    ds_emouerj = load_dataset_files('emouerj', file_path_emouerj)
    ds_mesd = load_dataset_files('mesd', file_path_mesd)
    ds_emovo = load_emovo_dataset(file_path_emovo)
    ds_oreau = load_oreau_dataset(file_path_oreau)

    train_sets, test_sets, val_sets = [], [], []
    
    for ds in [ds_iemocap, ds_subesco, ds_emodb, ds_emouerj, ds_emovo, ds_mesd, ds_oreau]:
        train_set, test_set, val_set = pp.split_dataset(ds)
        train_sets.append(train_set)
        test_sets.append(test_set)
        val_sets.append(val_set)

    ds_train_superset = concatenate_datasets(train_sets)
    ds_test_superset = concatenate_datasets(test_sets)
    ds_val_superset = concatenate_datasets(val_sets)

    ds_split = DatasetDict({
        'train': ds_train_superset,
        'validation': ds_val_superset,
        'test': ds_test_superset
    })

    return ds_split

def load_custom_subesco_superset(subesco_train_samples):
    ds_iemocap = load_iemocap()
    ds_emodb = load_dataset_files('emodb', file_path_emodb)
    ds_emouerj = load_dataset_files('emouerj', file_path_emouerj)
    ds_mesd = load_dataset_files('mesd', file_path_mesd)
    ds_emovo = load_emovo_dataset(file_path_emovo)
    ds_oreau = load_oreau_dataset(file_path_oreau)

    ds_subesco = load_subesco()

    indices = list(range(len(ds_subesco)))
    random.shuffle(indices)
    subset1 = ds_subesco.select(indices[:subesco_train_samples])

    # Sélectionnez le reste pour le second sous-ensemble
    subset2 = ds_subesco.select(indices[subesco_train_samples:])

    val, test = subset2.train_test_split(0.5)

    train_sets = [subset1, ds_iemocap, ds_emodb, ds_emouerj, ds_emovo, ds_mesd, ds_oreau]

    ds_train_superset = concatenate_datasets(train_sets)

    ds_split = DatasetDict({
        'train': ds_train_superset,
        'validation': val,
        'test': test
    })

    return ds_split


def load_superset_SUPERB():
    ds_iemocap = load_iemocap()
    ds_subesco = load_subesco()
    ds_emodb = load_dataset_files('emodb', file_path_emodb)
    ds_emouerj = load_dataset_files('emouerj', file_path_emouerj)
    ds_mesd = load_dataset_files('mesd', file_path_mesd)
    ds_emovo = load_emovo_dataset(file_path_emovo)
    ds_oreau = load_oreau_dataset(file_path_oreau)

    train, val, test = pp.split_dataset(ds_iemocap)

    train_sets = [train, ds_subesco, ds_emodb, ds_emouerj, ds_emovo, ds_mesd, ds_oreau]
    ds_train_superset = concatenate_datasets(train_sets)

    ds_split = DatasetDict({
        'train': ds_train_superset,
        'validation':val,
        'test': test
    })

    return ds_split