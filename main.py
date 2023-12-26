import load_datasets as ld
from Wav2Vec2MultiTask import Wav2Vec2MultiTask
import torch
import torch.nn as nn
from AudioDataset import AudioDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
import evaluation
import preprocess_data as pp
from train import train
import utils

PATH_SAVE = "i_love_my_model.pt"

def main():
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Utilisation de:", device)

    superset = ld.load_superset()

    superset = superset.map(pp.preprocess_function, remove_columns="audio", batched=True)

    train_dataset = AudioDataset(superset['train'])
    validation_dataset = AudioDataset(superset['validation'])
    test_dataset = AudioDataset(superset['test'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pp.collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True, collate_fn=pp.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=pp.collate_fn)

    label2id, id2label, num_labels = utils.create_label_mappings()

    # Choisir le model ici 
    model = Wav2Vec2MultiTask(7, num_labels=num_labels, label2id=label2id, id2label=id2label)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=15)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 15
    train(model, train_loader, validation_loader, optimizer, criterion, scheduler, num_epochs, device)

    torch.save(model.state_dict(), PATH_SAVE)

    accuracy, precision, recall, f1_score = evaluation.evaluate_model(model, test_loader, model.num_languages)
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}")

if __name__ == "__main__":
    main()
