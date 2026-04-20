from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.logger import get_logger

class FeatureEngineer(BaseEstimator,TransformerMixin):

    logger = get_logger(__name__)

    def __init__(self):     
        self.title_means = {}
        self._is_fitted = False 
    
    def _get_title(self,X):
        
        try:

            self.logger.info("Kötü yazılmış ünvanlar düzeltiliyor, Verisayısı yeterli olmayan ünvanlar aynı kategoriye alınıyor.")

            title = X['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

            title = title.replace('Mlle','Miss')
            title = title.replace('Ms','Miss')
            title = title.replace('Mme','Mrs')
            title = title.replace(['Countess','Jonkheer','Don','Dr','Rev','Lady','Major','Sir','Col','Capt'], 'Rare')

        except Exception as e:
            self.logger.error(f"Ünvanlar birleştirilemedi: {e}")
            raise


        return  title
    
    def fit(self, X, y=None):

        try:
            self.logger.info("Yaşların ortalaması alınıyor...")

            X_title = self._get_title(X)

            X_temp = (X.copy()
                    .assign(temp_title=X_title))
            
            self.title_means = (X_temp
                                .groupby('temp_title')['Age']
                                .mean()
                                .to_dict())
            
            self.logger.info("Yaşların ortalaması hesaplandı.")

            self._is_fitted = True
            return self

              
        except Exception as e:
            self.logger.error(f"Yaşların ortalaması alınamadı: {e}")
            raise

    def __sklearn_is_fitted__(self):  
        return self._is_fitted   
    
    def transform(self,X):
        try:
            self.logger.info("Feature engineering uygulanıyor..")

            X_copy = X.copy()
            X_copy['Title'] = self._get_title(X_copy)

            X_copy['Age'] = X_copy['Age'].fillna(X_copy['Title'].map(self.title_means))
            X_copy['Age'] = X_copy['Age'].round().astype('Int64')

            # FamilySize
            X_copy['FamilySize'] = X_copy['SibSp'] + X_copy['Parch'] + 1

            # IsAlone
            X_copy['IsAlone'] = (X_copy['FamilySize'] == 1).astype(int)

            X_copy = X_copy.drop(columns=['Name', 'SibSp', 'Parch'], errors='ignore')

            self.logger.info("Feature engineering tamamlandı.")
            return X_copy
        
        except Exception as e:
            self.logger.error(f"Feature engineering hatası: {e}")
            raise




