# heart-attack-prediction-service


### Background

Heart attacks kill 21 Australians every day [1]. The most common symptom is chest pain, however the presence of chest pain itself does not necessarily mean that a heart attack will follow. When combined with other observations, such as a patient's blood pressure, cholestoral level, and electrocardiographic (ECG) results, we can get a much better idea of the likelihood that a patient is at risk of suffering a heart attack.

This model will use demographic and clinical characteristics of patients presenting to hospital with chest pain, in order to identify those most at risk of suffering a heart attack, allowing them to receive potentially life-saving early treatment.


### About the [dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset):

- `age`: age of the patient
- `sex`: sex of the patient
- `exng`: exercise induced angina (1 = yes; 0 = no)
- `caa`: number of major vessels (0-3)
- `cp`: chest pain type
    - Value 1: typical angina
    - Value 2: atypical angina
    - Value 3: non-anginal pain
    - Value 4: asymptomatic
- `trtbps`: resting blood pressure (in mm Hg)
- `chol`: cholestoral in mg/dl fetched via BMI sensor
- `fbs`: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
- `restecg`: resting electrocardiographic results
    - Value 0: normal
    - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
- `thalachh`: maximum heart rate achieved
- `thall`: thalassemia
    - Value 0: null
    - Value 1: fixed defect
    - Value 2: normal
    - Value 3: reversable defect
- `oldpeak`: ST depression induced by exercise relative to rest
- `slp`: the slope of the peak exercise ST segment
- `output`: 
    - 0 = < 50% diameter narrowing; less chance of heart attack
    - 1 = > 50% diameter narrowing; more chance of heart attack


[1] https://www.healthdirect.gov.au/heart-attack


### Training


### Serving