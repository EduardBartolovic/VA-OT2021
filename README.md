# VA-OT2021

In Zeiten, in denen Neuronale Netze immer mehr Bedeutung erlangen ist es wichtig weiterhin die klassischen Algorithmen zu berücksichtigen, da diese oftmals ähnlich gute Ergebnisse erzielen. Im Rahmen der Veranstaltung Videoanalyse und Objekttracking wurden daher zwei mögliche Umsetzung für Objekt-Detektion und zwei für Objekt-Tracking näher betrachtet und implementiert. Jeweils eine der Umsetzung je Themenbereich arbeitet auf Basis von Neuronalen Netzen. Anschließend wurden für beide Varianten ein Objektzähler implementiert der die erkannten Objekte zählt. Ein wichtiger Bereich sind ebenfalls Fehler und Probleme der Algorithmen, welche anhand von aufgezeichneten Daten erläutert werden. Zuletzt wird ein Vergleich von klassischen Algorithmen mit Algorithmen auf Basis von Neuronalen Netzen gegeben.

## Was beinhaltet dieses Repository:

- Configuration für das Neuronale Netz YoloV4

- Gewichte für das vortrainierte Netz

- Namendatei für den COCO Datensatz

- Python-Sourcecode

(Original Repository für Coco, Configurations-Datei und Gewichte: https://github.com/AlexeyAB/darknet)

## Benötigte Bibliotheken:

- OpenCV
- TensorFlow

## Wie startet man den Code:

1. Projekt clonen
2. "Output" Ordner mit Unterordnern erstellen.
3. main.py anpassen an gewünschte Videodatei
4. main.py anpassen an gewünschte Klassen
5. main.py anpassen an gewünschten Algorithmus (classic, NN)
6. main.py starten
