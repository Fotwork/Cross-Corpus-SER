import torch.nn as nn
from transformers import Wav2Vec2Model
import torch
import utils

class Wav2Vec2MultiTask(nn.Module):
    def __init__(self, num_languages, num_labels, label2id, id2label, projector_dim=256, num_classes=4):
        super(Wav2Vec2MultiTask, self).__init__()
        self.num_languages = num_languages
        self.wav2vec_base = Wav2Vec2Model.from_pretrained(utils.MODEL_PATH, num_labels=num_labels, label2id=label2id, id2label=id2label)

        self.projector = nn.Linear(self.wav2vec_base.config.hidden_size, projector_dim)

        self.activation = nn.ReLU()

        # Têtes de classification pour chaque langue
        self.head1 = nn.Linear(projector_dim, num_classes)
        self.head2 = nn.Linear(projector_dim, num_classes)
        self.head3 = nn.Linear(projector_dim, num_classes)
        self.head4 = nn.Linear(projector_dim, num_classes)
        self.head5 = nn.Linear(projector_dim, num_classes)
        self.head6 = nn.Linear(projector_dim, num_classes)
        self.head7 = nn.Linear(projector_dim, num_classes)

    def forward(self, inputs, task_id):
        outputs = self.wav2vec_base(inputs).last_hidden_state
        average_output = torch.mean(outputs, dim=1)
        y = self.projector(average_output)
        act = self.activation(y)

        # Sélection de la tête de classification en fonction de task_id
        if task_id == 0:
            task_output = self.head1(act)
        elif task_id == 1:
            task_output = self.head2(act)
        elif task_id == 2:
            task_output = self.head3(act)
        elif task_id == 3:
            task_output = self.head4(act)
        elif task_id == 4:
            task_output = self.head5(act)
        elif task_id == 5:
            task_output = self.head6(act)
        elif task_id == 6:
            task_output = self.head7(act)
        return task_output