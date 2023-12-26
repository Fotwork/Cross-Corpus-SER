import torch

def train(model, train_loader, validation_loader, optimizer, criterion, scheduler, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0
        train_correct = 0
        train_total = 0

        for audio_samples, labels, language_ids in train_loader:
            audio_samples = audio_samples.to(device)
            labels = labels.to(device)
            language_ids = language_ids.to(device)


            optimizer.zero_grad()

            for task_id in range(model.num_languages):
                task_samples = audio_samples[language_ids == task_id]
                task_labels = labels[language_ids == task_id]

                if len(task_samples) == 0:
                    continue

                outputs = model(task_samples, task_id)

                loss = criterion(outputs, task_labels)
                loss.backward()
                optimizer.step()

                train_total_loss += loss.item()
                train_correct += (outputs.argmax(1) == task_labels).sum().item()
                train_total += task_labels.size(0)

        average_train_loss = train_total_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        print(f"Epoch {epoch+1}, Training Loss: {average_train_loss}, Training Accuracy: {train_accuracy}")

        # Étape de validation
        model.eval()
        validation_total_loss = 0
        validation_correct = 0
        validation_total = 0
        with torch.no_grad():
            for audio_samples, labels, language_ids in validation_loader:
                audio_samples = audio_samples.to(device)
                labels = labels.to(device)
                language_ids = language_ids.to(device)

                for task_id in range(model.num_languages):
                    task_samples = audio_samples[language_ids == task_id]
                    task_labels = labels[language_ids == task_id]

                    if len(task_samples) == 0:
                        continue

                    outputs = model(task_samples, task_id)
                    loss = criterion(outputs, task_labels)
                    validation_total_loss += loss.item()
                    validation_correct += (outputs.argmax(1) == task_labels).sum().item()
                    validation_total += task_labels.size(0)

        average_validation_loss = validation_total_loss / len(validation_loader)
        validation_accuracy = validation_correct / validation_total
        print(f"Epoch {epoch+1}, Validation Loss: {average_validation_loss}, Validation Accuracy: {validation_accuracy}")

        scheduler.step()