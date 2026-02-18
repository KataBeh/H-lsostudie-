# Hälsostudie- Del 2 

## Projektöversikt:
I del 2 av projektet fortsätter jag arbete från del 1 och bygger vidare med:
- modulstruktur i mappen scr/
- en egen klass (HealthAnalyzer) för koder och visualiseringar
- linjär regression med scikit-learn
- nya grafer och tolkningar
- PCA för att hitta mönster i datan
- dokumentation med markdowns 
- reproducerbar notebook (Restart & Run All fungerar utan fel)

## Projektstruktur:

Hälsostudie/
│
├── data/
│   └── health_study_dataset.csv
│
├── scr/
│   └── health_analyzer.py       # Klass med analysfunktioner
│
├── notebook_del1.ipynb          # Del 1 
├── notebook_del2.ipynb          # Del 2 
│
├── requirements.txt             # Paket jag använder
├── .gitignore                   # .venv exkluderas
└── README.md

## Syfte:
Målet med Del2 är att fördjupa analysen som jag tidigare byggde genom att strukturera koden bättre, skapa en klass, göra fler visualiseringar och testa linjär regression i praktiken samt försöka hitta ett mönster i datan med hjälp av PCA. 

## Sammanfattning baserat på aktuell data:
I del 2 har jag:
- Strukturerat om koden genom att skapa en klass 'HealthAnalyzer' i en egen modul.
- Flytat delar av analysen från del 1 till metoder i klassen.
- Skapat en korrelationsmatris för att se vilka variabler påverkar varandra. 
- Byggt en linjär regressionsmodell med 'LinearRegression' från scikit-learn för at förutsäga systoliskt blodtryck utifrån ålder och vikt.
- Undersökt sambandet mellan ålder och blodtryck med ett scatterplot.
- Jämfört sjukdomsförekomst per kön med ett stapeldiagram.
- Lagt till en **avancerad analys med PCA** som används för att hitta mönster i hälsodatan baserat på flera variabler samtidigt( vanligt inom medicinsk forskning, bildanalys och AI, alltså data med många variabler).
- Reducerade datan till PC1 och PC2 vilket visualiserades i scatterplott. Det är lite mer avancerat än enkel regression, men den metoden gör lättare att visualisera strukturen.
- PCA analysen visade att datan är **komplex** (ganska spridd) och hälsan påverkas av flera faktorer- inte bara en. 

## Miljö
- **Python:** 3.13.7
- **Paket:** `Pandas`, `matplotlib`, `Numpy`

## Hur man kör projektet:

```bash
# klona projektet
git clone https://github.com/KataBeh/H-lsostudie-.git

cd H-lsostudie-

# Skapa och aktivera virtuell miljö
python -m venv . venv
# Windows PowerShell
.venv\Scripts\Activate
# Installera paket:
pip install -r requirements.txt
# Öppna notebook_del2.ipynb och kör Run All
```

## Hur man använder klassen:
```python
from scr.health_analyzer import HealthAnalyzer
import pandas as pd

df = pd.read_csv("data/health_study_dataset.csv")
analyzer = HealthAnalyzer(df)

# Kör analyser:
analyzer.chech_data_quality()
analyzer.basic_info(cols)
analyzer.plot_bp_histogram()
analyzer.show_correlations()
analyzer.plot_age_vs_bp()
analyzer.plot_disease_rate_by_sex()
analyzer.plot_pca()
```

## Slutsats:
- Ålder och vikt har viss påverkan på blodtrycket (R² = ~0.40).
- Män hade något högre sjukdomsförekomst än kvinnor.
- PCA visar att hälsa påverkas av **flera variabler samtidigt**, vilket betyder att datan är komplex och inte kan förklaras av en faktor. 
