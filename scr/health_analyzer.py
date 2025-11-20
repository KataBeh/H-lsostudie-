import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


class HealthAnalyzer:
    """
    Analyserar data från Hälsostudien.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def check_data_quality(self) -> None:
        """
        Kollar saknade värden + duplicerade rader.
        """
        print("Saknade värden per kolumn:")
        print(self.df.isna().sum())

        print("\nFinns duplicerade rader?:")
        print(self.df.duplicated().any())


    def basic_info(self, cols: list[str]) -> pd.DataFrame:
        """
        Beräknar grundstatistik (medel, median, min, max) för valda kolumner.
        """
        return self.df[cols].agg(["mean", "median", "min", "max"])
    
    def plot_bp_histogram(self, bins: int = 20) -> None:
        """
        Ritar ett histogram över systoliskt blodtryck.
        """
        plt.figure(figsize=(8, 5))
        plt.hist(
            self.df["systolic_bp"], 
            bins = bins,
            label= "Fördelning av blodtryck",
            color="skyblue",
            edgecolor="black"
        )
        plt.xlabel("Systoliskt blodtryck (mmHg)")
        plt.ylabel("Antal personer")
        plt.title("Fördelning av systoliskt blodtryck")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.show()


    # Vill se vilka variabler som påverkar varandra:
    def show_correlations(self, cols=None):
        """
        Visar en korrelationsmatris för att se hur variabler hänger ihop. Perfekt innan regression/PCA modellerna.
        """
        if cols is None:
            cols = ["age", "weight", "height", "systolic_bp", "cholesterol"]
        
        corr = self.df[cols].corr()
        print(corr)

        plt.figure(figsize=(8, 5))
        plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
        plt.colorbar(label="Korrelationsvärde")
        plt.title("Korrelationsmatris")
        plt.xticks(range(len(cols)), cols, rotation=45)
        plt.yticks(range(len(cols)), cols)
        plt.show()


    # använder "fit" som betyder att den här funktionen tränar modellen på datan; analys av relationen mellan blodtryck och ålder:
    def fit_bp_regression(self, features: list[str] | None = None):
        """
          Tränar en linjär regressionsmodell som försöker förutsäga blodtryck från ålder och vikt.
          Modellen använder ålder, vikt eller andra kolumner som förklarar variabler (x) och systoliskt blodtryck som
          det andra värde vi vill förutsäga (y).

         Om inget annat anges används ['age', 'weight'] som standard. 

         float 
            Modellens R^2-värde (hur bra modellen passar datan)
         """

        if features is None:
           features= ["age", "weight"]

        x = self.df[features].values
        y = self.df["systolic_bp"].values

        model = LinearRegression()
        model.fit(x, y)

        r2 = model.score(x, y)

        self.bp_model = model
        self.bp_features = features

        return model, r2
    

    # ritar en graf till sambandet mellan ålder och blodtryck:
    def plot_age_vs_bp(self) -> None:
        """
        Ritar en scatter som visar sambandet mellan ålder och systoliskt blodtryck.
        """
        plt.figure(figsize=(8, 5))
        plt.scatter(
            self.df["age"],
            self.df["systolic_bp"],
            alpha = 0.6,
            color = "teal",
            edgecolor = "black"
        )
        plt.xlabel("Ålder")
        plt.ylabel("Blodtryck (mmHg)")
        plt.title("Samband mellan ålder och blodtryck")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    # analys av sjukdomsförekomst per kön i en stapeldiagram:
    def plot_disease_rate_by_sex(self) -> None:
        """
        Ritar upp en stapeldiagram, där man ser andelen personer med sjukdom uppdelar per kön.
        """
        rates = self.df.groupby("sex")["disease"].mean() #mean på 0/1-kolumn=andel
        mean_rate = rates.mean()

        plt.figure(figsize=(8, 5))
        rates.plot(kind="bar", color=["skyblue", "lightcoral"], edgecolor="black")
        plt.axhline(mean_rate, color="gray", linestyle="--", label= "Genomsnittet av båda grupperna", alpha=0.6)
        plt.ylabel("Andel med sjukdom")
        plt.ylim(0, 1)
        plt.title("Sjukdomsförekomst per kön")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend()
        plt.show()


    # försöker hitta mönster i datan:
    def pca_analysis(self, features=None, n_components=2):
        """
        Använder PCA på vålda kolumner för att hitta mönster i datan.
        """
        if features is None:
            features = ["age", "weight", "height", "cholesterol", "systolic_bp"]
            X = self.df[features].values

            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(X)

        return transformed, pca.explained_variance_ratio_
    

    # Ritar upp en graf som visualiserar PCA metoden lite bättre:
    def plot_pca(self, features=None):
        """
        Ritar en scatterplot med de två första PCA-komponenterna. Lättare att se visuellt ett mönster eller grupper i datan.
        """
        transformed, variance = self.pca_analysis(features)

        plt.figure(figsize=(8, 6))
        plt.scatter(transformed[:,0], transformed[:,1], alpha=0.6, edgecolor="black")
        plt.xlabel(f"PC1 ({variance[0]*100: .1f}%)")
        plt.ylabel(f"PC2 ({variance[1]*100: .1f}%)")
        plt.title("PCA - Mönster i hälsodatan")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

