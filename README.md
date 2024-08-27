# Contrastive Learning mit Stable Diffusion-basierter Datenaugmentation: Verbesserung der Bildklassifikation durch synthetische Daten

Teil einer Bachelorarbeit im Studiengang Medientechnik an der Hochschule für Angewandte Wissenschaften Hamburg.

Mit dieser Methode wird ein [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)-Klassifikator trainiert, der automatisch mit [DA-Fusion](https://arxiv.org/abs/2302.07944) synthetische Augmentationen generiert und ins Training integriert.

## Hintergrund

...

## Installation

...

## Anwendung

- Datensatz-Klasse (datasets.py)

da-fusion
    output
        experiment_name
            extracted
            fine-tuned
            fine-tuned-merged
sup-contrast
    output
        experiment_name
            models

...

- [DA-Fusion](da-fusion/README.md)
- [Supervised Contrastive Learning](sup-contrast/README.md)