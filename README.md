Bachelorarbeit im Studiengang Medientechnik an der [HAW Hamburg](https://www.haw-hamburg.de/)
# Contrastive Learning mit Stable Diffusion-basierter Datenaugmentation: Verbesserung der Bildklassifikation durch synthetische Daten

Es wurde ein [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)-Klassifikator trainiert, der synthetische Daten verwendet, welche zuvor mit [DA-Fusion](https://arxiv.org/abs/2302.07944) generiert wurden. DA-Fusion ist eine Methode zur Stable Diffusion-basierten Datenaugmentation, welche semantisch sinnvolle Variationen von Bildern generieren kann.

Mit DA-Fusion wurden sowohl In-Distribution- also auch (Near) Out-of-Distribution-Daten generiert, indem die Stärke der Augmentation unterschiedlich eingestellt wurde. Die OOD-Daten sollten dabei nur als negativ-Beispiele im Contrastive Learning dienen, um die Repräsentationen der ID-Daten weiter zu verbessern. Die Experimente der Arbeit zeigten, dass die synthetischen ID-Daten zu einer Verbesserung der Klassifikation beitragen, die OOD-Daten jedoch nicht.

## Installation

```bash
conda create -n synt-contrast python=3.7 pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6
conda activate synt-contrast
pip install diffusers["torch"] transformers pycocotools pandas matplotlib seaborn scipy
pip install -e da-fusion
pip install --upgrade huggingface_hub
huggingface-cli login
```

(Conda-Channels: `nvidia`, `pytorch`, `conda-forge`)

## Verwendung

Vollständige Pipelines für den [MVIP](https://fordatis.fraunhofer.de/handle/fordatis/358)-Datensatz:

- `mvip_generate.augs.sh` zur Generierung synthetischer ID- & OOD-Augmentationen
- `mvip_run_experiments.sh` zum Ausführen drei unterschiedlicher Trainingsdurchläufe mit Supervised Contrastive Learning, um den Einfluss der Augmentationen auf die Klassifikation zu untersuchen

Die Ausführlichen READMEs zu den beiden verwendeten Methoden:

- [DA-Fusion](da_fusion/README.md)
- [Supervised Contrastive Learning](sup_contrast/README.md)