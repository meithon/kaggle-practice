"""「女性なら生存、男性なら死亡」ルールの予測器。"""

import pandas as pd

from predictor import Predictor


class FemaleRule(Predictor):
    def predict(self, df: pd.DataFrame) -> pd.Series:
        return (df["Sex"].str.lower() == "female").astype(int)
