import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.logger import get_logger


class Cleaner(BaseEstimator,TransformerMixin):

    logger = get_logger(__name__)

    def __init__(self, drop_cols=None):

        if drop_cols is None:
            self.drop_cols = ['PassengerId', 'Ticket', 'Cabin']
        else:
            self.drop_cols = drop_cols

        self._is_fitted = False     


    def fit(self,X,y=None): 
        self._is_fitted = True
        return self # Temizlik işleminde 'öğrenilecek' bir şey olmadığı için sadece self döner
    
    def __sklearn_is_fitted__(self):  # ← ekle
        return self._is_fitted
    
    def transform(self,X):

        try:
            self.logger.info("Transfor metotu başlatılıyor: ")

            X_copy = X.copy()

            X_copy = X_copy.drop(
                columns = [
                    col for col in self.drop_cols
                    if col in X_copy.columns
                          ]
                )
           
            if 'Embarked' in X_copy.columns:
                X_copy['Embarked'] = X_copy['Embarked'].fillna('C')

            if 'Fare' in X_copy.columns:
                X_copy['Fare'] = X_copy['Fare'].fillna(X_copy['Fare'].median())    
            
            return X_copy
        
        except Exception as e:
            self.logger.error(f"Transform sırasında hata oluştu: {e}")

       