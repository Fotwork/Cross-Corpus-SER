# Choisir le model ici 
MODEL_PATH = "facebook/wav2vec2-base-960h"

def create_label_mappings():
    labels = {
        "neutral": 0,
        "happy": 1,
        "angry": 2,
        "sad": 3
    }
    label2id = {label: str(id) for label, id in labels.items()}
    id2label = {str(id): label for label, id in labels.items()}
    return label2id, id2label, len(id2label)