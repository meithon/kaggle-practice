"""予測器の抽象基底クラス。predict の実装を強制する。"""

from abc import ABC, abstractmethod

import pandas as pd


class Predictor(ABC):
    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """入力 DataFrame を受け取り、Survived の予測 (0/1) を返す。"""
