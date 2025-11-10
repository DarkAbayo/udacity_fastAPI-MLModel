# ProjektDescription : Bereitstellung eines Machine Learning Modells mit FastAPI

## Projektanleitung

### Arbeitsumgebung
- Arbeiten in einer Kommandozeilenumgebung wird empfohlen, um die Nutzung von Git und DVC zu erleichtern. Bei Windows wird WSL1 oder 2 empfohlen.

### Repositories
- Erstelle ein Verzeichnis für das Projekt und initialisiere Git.
  - Während du am Code arbeitest, committe kontinuierlich Änderungen. Trainierte Modelle, die du behalten möchtest, müssen in GitHub committed werden.
- Verbinde dein lokales Git-Repository mit GitHub.

### GitHub Actions
- Richte GitHub Actions in deinem Repository ein. Du kannst eine der vorgefertigten GitHub Actions verwenden, die mindestens `pytest` und `flake8` bei jedem Push ausführt und beide ohne Fehler bestehen müssen.
  - Stelle sicher, dass die GitHub Action die gleiche Python-Version verwendet, die du in der Entwicklung genutzt hast.

### Daten
- Lade `census.csv` aus dem Datenordner im Starter-Repository herunter.
  - Informationen zum Datensatz findest du [hier](https://archive.ics.uci.edu/ml/datasets/census+income).
- Diese Daten sind unordentlich, versuche sie in Pandas zu öffnen und zu sehen, was du bekommst.
- Um sie zu bereinigen, benutze deinen bevorzugten Texteditor, um alle Leerzeichen zu entfernen.

### Modell
- Verwende den Startercode, um ein Machine Learning Modell zu schreiben, das auf den bereinigten Daten trainiert und das Modell speichert. Vervollständige jede Funktion, die begonnen wurde.
- Schreibe Unit-Tests für mindestens 3 Funktionen im Modellcode.
- Schreibe eine Funktion, die die Leistung des Modells auf Daten-Slices ausgibt.
  - Vorschlag: Zur Vereinfachung kann die Funktion einfach die Leistung auf Slices der kategorialen Merkmale ausgeben.
- Schreibe eine Modellkarte mit der bereitgestellten Vorlage.

### API-Erstellung
- Erstelle eine RESTful API mit FastAPI, die Folgendes implementiert:
  - GET auf der Wurzel, die eine Willkommensnachricht gibt.
  - POST, die Modellinferenz durchführt.
  - Typ-Hinweise müssen verwendet werden.
  - Verwende ein Pydantic-Modell, um den Body von POST zu verarbeiten. Dieses Modell sollte ein Beispiel enthalten.
    - Hinweis: Die Daten haben Namen mit Bindestrichen, und Python erlaubt diese nicht als Variablennamen. Ändere die Spaltennamen in der CSV nicht und verwende stattdessen die Funktionalität von FastAPI/Pydantic/etc., um damit umzugehen.
- Schreibe 3 Unit-Tests, um die API zu testen (einen für GET und zwei für POST, einen für jede Vorhersage).
- **Führe einen Sanity-Check für deine Testfälle durch:**
  - Führe `python sanitycheck.py` aus. Dieses Skript befindet sich im Starterverzeichnis im Startercode.
  - Das Skript scannt die geschriebenen Testfälle für die `GET()` und `POST()` APIs und generiert einen Bericht.
  - Der Bericht listet alle Probleme auf, die es erkennt. Behebe die Probleme und führe das Skript `sanitycheck.py` erneut aus.
  - Das Skript verwendet Heuristiken, um häufige Probleme zu erkennen und kann manchmal ein Problem übersehen oder einen Fehlalarm auslösen. Du solltest dennoch deine Implementierung anhand der Projekt-Rubrik überprüfen, um sicherzustellen, dass deine Einreichung die Anforderungen erfüllt.

### Bereitstellung der API auf einer Cloud-Anwendungsplattform
- Erstelle ein Konto bei einer Cloud-Anwendungsplattform, wie `Heroku` oder `Render`. Für die folgenden Schritte kannst du entweder die Web-GUI verwenden oder die CLI herunterladen.
  - *Hinweis: Ab dem 28. November 2022 hat Heroku die Entfernung des kostenlosen Kontos angekündigt, das durch einen kostengünstigen Abonnementplan ersetzt wird. (Referenz: [Entfernung der Heroku Free Product Plans FAQ](https://help.heroku.com/RSBRUH58/removal-of-heroku-free-product-plans-faq))*
- Erstelle eine neue App und lasse sie aus deinem GitHub-Repository bereitstellen.
  - Aktiviere automatische Bereitstellungen, die nur bereitgestellt werden, wenn deine kontinuierliche Integration besteht.
  - Hinweis: Denke daran, wie sich die Pfade in deiner lokalen Umgebung von denen auf Heroku unterscheiden.
  - Hinweis: Die Entwicklung in Python ist schnell! Aber wie schnell du iterieren kannst, verlangsamt sich, wenn du darauf angewiesen bist, dass deine CI/CD fehlschlägt, bevor du ein Problem behebst. Ich empfehle, `flake8` lokal auszuführen, bevor du Änderungen committest.
- Schreibe ein Skript, das das `requests`-Modul verwendet, um einen POST auf deiner Live-API durchzuführen.

## Projekt-Rubrik

### Git
- **Kriterium:** Git mit GitHub Actions einrichten.
  - **Bestanden:** 
    - GitHub Action sollte `pytest` und `flake8` bei jedem Push auf `main/master` ausführen.
    - `PyTest` muss bestehen (bis das Projekt abgeschlossen ist, sollten mindestens sechs Tests vorhanden sein) und `flake8` muss ohne Fehler bestehen.
    - Füge entweder einen Link zu deinem GitHub-Repository oder einen Screenshot der erfolgreichen CI mit dem Namen `continuous_integration.png` hinzu.

### Modellbau
- **Kriterium:** Erstelle ein Machine Learning Modell.
  - **Bestanden:** 
    - Das Modell sollte auf den bereitgestellten Daten trainiert werden. Die Daten sollten entweder aufgeteilt werden, um einen Train-Test-Split zu haben, oder es sollte eine Kreuzvalidierung auf dem gesamten Datensatz durchgeführt werden.
    - Implementiere alle stubbed Funktionen im Startercode oder erstelle Äquivalente. Mindestens sollten Funktionen vorhanden sein, um:
      - das Modell zu trainieren, zu speichern und zu laden sowie alle kategorialen Encoder.
      - Modellinferenz.
      - die Klassifikationsmetriken zu bestimmen.
    - Schreibe ein Skript, das die Daten einliest, verarbeitet, das Modell trainiert und es sowie den Encoder speichert. Dieses Skript muss die Funktionen verwenden, die du geschrieben hast.

- **Kriterium:** Schreibe Unit-Tests.
  - **Bestanden:** 
    - Schreibe mindestens 3 Unit-Tests. Unit-Tests für ML können schwierig sein, da sie stochastisch sind – teste mindestens, ob ML-Funktionen den erwarteten Typ zurückgeben.

- **Kriterium:** Schreibe eine Funktion, die Modellmetriken auf Daten-Slices berechnet.
  - **Bestanden:** 
    - Schreibe eine Funktion, die die Leistung auf Modell-Slices berechnet. D.h. eine Funktion, die die Leistungsmetriken berechnet, wenn der Wert eines bestimmten Merkmals festgelegt ist. Z.B. für Bildung würde es die Modellmetriken für jeden Slice von Daten ausgeben, der einen bestimmten Wert für Bildung hat. Du solltest einen Satz von Ausgaben für jeden einzelnen einzigartigen Wert in der Bildung haben.
    - Vervollständige die stubbed Funktion oder schreibe eine neue, die für eine gegebene kategoriale Variable die Metriken berechnet, wenn ihr Wert festgelegt ist.
    - Schreibe ein Skript, das diese Funktion ausführt (oder füge sie als Teil des Trainingsskripts hinzu), das durch die unterschiedlichen Werte eines der Merkmale iteriert und die Modellmetriken für jeden Wert ausgibt.
    - Gib die Ausgabe in einer Datei mit dem Namen `slice_output.txt` aus.

- **Kriterium:** Schreibe eine Modellkarte.
  - **Bestanden:** 
    - Die Modellkarte sollte jeden Abschnitt der Vorlage ansprechen. **Bitte verwende die bereitgestellte [Modellkarten-Vorlage](https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/model_card_template.md).**
    - Die Modellkarte sollte in vollständigen Sätzen geschrieben sein und Metriken zur Modellleistung enthalten. Bitte füge sowohl die verwendeten Metriken als auch die Leistung deines Modells bei diesen Metriken hinzu.

### API-Erstellung
- **Kriterium:** Erstelle eine REST API.
  - **Bestanden:** 
    - Die API muss GET und POST implementieren. GET muss auf der Wurzel-Domain sein und eine Begrüßung ausgeben, und POST auf einem anderen Pfad, der Modellinferenz durchführt.
    - Verwende Python-Typ-Hinweise, sodass FastAPI die automatische Dokumentation erstellt.
    - Verwende ein Pydantic-Modell, um den Body des POST zu verarbeiten. Dies sollte ein Beispiel implementieren (Hinweis: Pydantic/FastAPI bietet mehrere Möglichkeiten, dies zu tun, siehe die Dokumentation: https://fastapi.tiangolo.com/tutorial/schema-extra-example/).
    - Füge einen Screenshot der Dokumentation hinzu, der das Beispiel zeigt, und nenne ihn `example.png`.

- **Kriterium:** Erstelle Tests für eine API.
  - **Bestanden:** 
    - Du solltest mindestens drei Testfälle schreiben - 
      - Ein Testfall für die GET-Methode. Dies MUSS sowohl den Statuscode als auch den Inhalt des Anforderungsobjekts testen.
      - Ein Testfall für JEDEN der möglichen Inferenz (Ergebnisse/Ausgaben) des ML-Modells.

### API-Bereitstellung
- **Kriterium:** Stelle eine App auf einer Cloud-Anwendungsplattform bereit.
  - **Bestanden:** 
    - Bereitstellung über ein GitHub-Repository mit aktivierter kontinuierlicher Bereitstellung. Füge einen Screenshot hinzu, der zeigt, dass CD aktiviert ist und nenne ihn `continuous_deployment.png`.
    - Füge einen Screenshot deines Browsers hinzu, der den Inhalt des GET anzeigt, den du auf der Wurzel-Domain implementiert hast. Nenne diesen Screenshot `live_get.png`.

- **Kriterium:** Abfrage der Live-API.
  - **Bestanden:** 
    - Schreibe ein Skript, das POST an die API unter Verwendung des `requests`-Moduls sendet und sowohl das Ergebnis der Modellinferenz als auch den Statuscode zurückgibt. Füge einen Screenshot des Ergebnisses hinzu. Nenne diesen `live_post.png`.
