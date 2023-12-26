# Reconnaissance Multilingue des Émotions dans la Parole avec wav2vec2 et HuBERT

Ce projet vise à reconnaître les émotions dans des enregistrements vocaux multilingues en utilisant les modèles wav2vec2 et HuBERT. Il combine des techniques de traitement du langage naturel et d'apprentissage profond pour analyser et comprendre les émotions exprimées dans la parole.

## Structure du Projet

Le projet est organisé en plusieurs fichiers Python, chacun ayant un rôle spécifique :

- `load_datasets.py` : Contient des méthodes pour charger les datasets nécessaires pour l'entraînement et l'évaluation du modèle.
- `preprocess_data.py` : Utilisé pour le prétraitement des données audio, y compris la normalisation, le découpage, etc.
- `utils.py` : Contient des fonctions utilitaires utilisées à travers le projet.
- `train.py` : Script principal pour l'entraînement du modèle.
- `Wav2Vec2MultiTask.py` : Définit la structure du modèle wav2vec2 multi-tache pour la reconnaissance des émotions.
- `evaluation.py` : Méthodes pour évaluer le modèle sur un ensemble de données de test.
- `AudioDataset.py` : Classe personnalisée pour gérer les ensembles de données audio.
- `main.py` : Fichier principal qui orchestre le processus complet, de la charge des données à l'évaluation du modèle.

## Reproduction des Expériences

Pour reproduire les expériences menées dans ce projet, suivez simplement les instructions et le template fournis dans le fichier `main.py`. Ce template est conçu pour guider l'utilisateur à travers les différentes étapes du processus, de la préparation des données à l'évaluation du modèle. Il suffit juste de loader le dataset sur lequel vous voulez experimenter parmis toutes les methodes dans load_datasets.py.

## Configuration Requise

- Python 3.x
- Bibliothèques (CALCUL QUEBEC): 
    ```
    pip install accelerate -U 
    pip install transformers[torch] 
    module load gcc/9.3.0 arrow/11.0.0 
    pip install --no-index datasets 
    pip install librosa
    ```

## Choix du Modèle

Pour choisir le modèle à utiliser, modifiez le chemin dans `utils.py`
