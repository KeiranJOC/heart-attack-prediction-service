import bentoml

from bentoml.io import JSON
from pydantic import BaseModel


# define Pydantic class to validate input

class PatientFeatures(BaseModel):
    age: int = 63
    sex: int = 1
    cp: int = 3
    trtbps: int = 145
    chol: int = 233
    fbs: int = 1
    restecg: int = 0
    thalachh: int = 150
    exng: int = 0
    oldpeak: float = 2.3
    slp: int = 0
    caa: int = 0
    thall: int = 1


model_ref = bentoml.xgboost.get('heart_attack_prediction_model:latest')
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service('heart_attack_prediction_model', runners=[model_runner])


@svc.api(input=JSON(pydantic_model=PatientFeatures), output=JSON())
def classify(patient_features):
    application_data = patient_features.dict()
    vector = dv.transform(application_data)
    prediction = model_runner.predict.run(vector)
    result = prediction[0]

    print(result)

    if result > 0.5:
        return { "Risk of heart attack": "LIKELY" }
    else:
        return { "Risk of heart attack": "UNLIKELY" }
    