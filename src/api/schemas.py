from pydantic import BaseModel

class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    Title: str = "Mr" # Feature engineering'den gelen default değer

class PredictionResponse(BaseModel):
    passenger_id: int
    survived: int
    probability: float