Bachelorarbeit im Studiengang Medientechnik an der [HAW Hamburg](https://www.haw-hamburg.de/)
# Contrastive Learning mit Stable Diffusion-basierter Datenaugmentation: Verbesserung der Bildklassifikation durch synthetische Daten

Mit dieser Methode wird ein [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)-Klassifikator trainiert, der automatisch mit [DA-Fusion](https://arxiv.org/abs/2302.07944) synthetische Augmentationen generiert und ins Training integriert. Es werden sowohl In-Distribution also auch Out-of-Distribution Daten generiert. Die OOD-Daten werden dabei nur als negativ-Beispiele im Contrastive Learning verwendet, um die Generalisierung und Robustheit des Modells insgesamt zu verbessern.

## Hintergrund

...

## Installation

```bash
conda create -n synt-contrast python=3.7 pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c nvidia -c pytorch -c conda-forge
conda activate synt-contrast
pip install diffusers["torch"] transformers pycocotools pandas matplotlib seaborn scipy
pip install -e da-fusion
pip install --upgrade huggingface_hub
huggingface-cli login
```

## Verwendung

- [DA-Fusion](da-fusion/README.md)
- [Supervised Contrastive Learning](sup-contrast/README.md)