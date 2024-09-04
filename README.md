Bachelorarbeit im Studiengang Medientechnik an der [HAW Hamburg](https://www.haw-hamburg.de/)
# Contrastive Learning mit Stable Diffusion-basierter Datenaugmentation: Verbesserung der Bildklassifikation durch synthetische Daten

Mit dieser Methode wird ein [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)-Klassifikator trainiert, der mit [DA-Fusion](https://arxiv.org/abs/2302.07944) generierte synthetische Augmentationen Training integriert. Es können sowohl In-Distribution- also auch Out-of-Distribution-Daten generiert und verwendet werden. Die OOD-Daten dienen dabei nur als negativ-Beispiele im Contrastive Learning. Die Experimente dieser Arbeit zeigten, dass die synthetischen ID-Daten zu einer Verbesserung der Klassifikation beitragen, die OOD-Daten jedoch nicht.

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

Pipelines:

- `mvip_generate.augs.sh` zur Generierung synthetischer ID- & OOD-Augmentationen
- `mvip_run_experiments.sh` zum Ausführen drei unterschiedlicher Trainingsdurchläufe mit Supervised Contrastive Learning, um den Einfluss der Augmentationen auf die Klassifikation zu untersuchen

Die Ausführlichen READMEs:

- [DA-Fusion](da-fusion/README.md)
- [Supervised Contrastive Learning](sup-contrast/README.md)