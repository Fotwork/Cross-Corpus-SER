from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch

def evaluate_model(model, test_loader, num_languages):
    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for audio_samples, labels, language_ids in test_loader:
            for task_id in range(num_languages):
                task_samples = audio_samples[language_ids == task_id]
                task_labels = labels[language_ids == task_id]

                if len(task_samples) == 0:
                    continue

                outputs = model(task_samples, task_id)
                all_outputs.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(task_labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_outputs)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_outputs, average='macro')

    return accuracy, precision, recall, f1_score