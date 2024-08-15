# Contrastive Learning mit Stable Diffusion-basierter Datenaugmentation: Verbesserung der Bildklassifikation durch synthetische Daten

Diese Implementierung ist Teil einer Bachelorarbeit im Studiengang Medientechnik an der Hochschule für Angewandte Wissenschaften Hamburg.

Ziel ist die Implmentierung einer [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)-Methode, welche [DA-Fusion](https://arxiv.org/abs/2302.07944) zur synthetischen Datenaugmentation verwendet. Mit der Kombination soll für eine Bildklassifikations-Aufgabe eine verbesserte Generalisierungsfähigkeit und Robustheit gegenüber Out-of-Distribution-Daten erzielt werden.

## Hintergrund

...

## Installation & Verwendung

Die Implementierung besteht aus zwei separaten Anwendung, die hintereinander ausgeführt werden - einmal die synthetische Datenaugmentation mit DA-Fusion, und anschließend das Training und die Bildklassifikation mit Supervised Contrastive Learning. Die Installation, das Setup, und die Verwendung mit eigenen Datensätzen werden in den jeweiligen READMEs separat beschrieben:

- [DA-Fusion](da-fusion/README.md)
- [Supervised Contrastive Learning](sup-contrast/README.md)