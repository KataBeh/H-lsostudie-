import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class HealthAnalyzer:
    """
    Analyserar data från Hälsostudien.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def basic_info(self, cols: list[str]) -> pd.DataFrame:
        """
        Beräknar grundnstatistik (medel, median, min, max) för valda kolumner.
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


    # använder "fit" som betyder att den här funktionen tränar modellen på datan
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