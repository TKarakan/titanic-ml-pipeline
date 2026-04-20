from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.logger import get_logger
import pandas as pd

logger = get_logger(__name__)


class Encoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.embarked_columns = []
        self.title_columns = []
        self._is_fitted = False

    def fit(self, X, y=None):
        """
        Sadece öğren — hangi kategoriler var sakla.
        Dönüştürme yapma.
        """
        try:
            logger.info("Encoder fit ediliyor...")

            # Embarked'daki tüm unique değerleri öğren
            embarked_dummies = pd.get_dummies(X['Embarked'], prefix='Embarked')
            self.embarked_columns = embarked_dummies.columns.tolist()

            # Title'daki tüm unique değerleri öğren
            title_dummies = pd.get_dummies(X['Title'], prefix='Title')
            self.title_columns = title_dummies.columns.tolist()

            logger.info(f"Embarked sütunları: {self.embarked_columns}")
            logger.info(f"Title sütunları: {self.title_columns}")
            self._is_fitted = True
            return self  # fit her zaman self döner

        except Exception as e:
            logger.error(f"Encoder fit hatası: {e}")
            raise

    def transform(self, X):
        """
        Öğrendiklerini uygula.
        fit'te saklanan sütun listelerini kullan.
        """
        try:
            logger.info("Encoding uygulanıyor...")
            X_copy = X.copy()

            # 1. Sex → binary
            X_copy['Sex'] = (X_copy['Sex'] == 'male').astype(int)

            # 2. Embarked → one-hot
            embarked_dummies = pd.get_dummies(X_copy['Embarked'], prefix='Embarked')
            # reindex → fit'te görülen sütunları garanti et
            # eksik sütun → 0, fazla sütun → çıkar
            embarked_dummies = embarked_dummies.reindex(
                columns=self.embarked_columns, fill_value=0
            )
            X_copy = X_copy.drop(columns=['Embarked'])
            X_copy = pd.concat([X_copy, embarked_dummies], axis=1)

            # 3. Title → one-hot, aynı mantık
            title_dummies = pd.get_dummies(X_copy['Title'], prefix='Title')
            title_dummies = title_dummies.reindex(
                columns=self.title_columns, fill_value=0
            )
            X_copy = X_copy.drop(columns=['Title'])
            X_copy = pd.concat([X_copy, title_dummies], axis=1)

            logger.info("Encoding tamamlandı.")
            return X_copy

        except Exception as e:
            logger.error(f"Encoding hatası: {e}")
            raise

    def __sklearn_is_fitted__(self):  # ← ekle
        return self._is_fitted    