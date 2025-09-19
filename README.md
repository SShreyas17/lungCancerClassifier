# Lung Cancer Classifier

An AI-powered application that uses deep learning on CT scan images to classify lung cancer types with an interactive GUI.



## Repository Structure
```
lungCancerClassifier/
├── data/                             # Dataset used to train the model
│   ├──test/
│   │   ├──adenocarcinoma/
│   │   ├──large.cell.carcinoma/
│   │   ├──normal/
│   │   └──squamous.cell.carcinoma/
│   ├──train/
│   │   ├──adenocarcinoma/
│   │   ├──large.cell.carcinoma/
│   │   ├──normal/
│   │   └──squamous.cell.carcinoma/
│   └──valid/
│       ├──adenocarcinoma/
│       ├──large.cell.carcinoma/
│       ├──normal/
│       └──squamous.cell.carcinoma/
├── notebooks/                        # Additional documentation
│   ├── model_training.ipynb
│   └── model_evaluation.ipynb
├── app.py                            # GUI application script
├── requirements.txt                  # Dependencies
├── README.md                         # Project overview and instructions
└── LICENSE                           # License file
```

## Models

We provide two versions each of **DenseNet201** and **ResNet50**:
1. Original pretrained models  
2. Custom fine-tuned versions (with hyperparameter tuning)

These models are **not stored in the repo**.  
You can download them from Google Drive:

[Download Models (Google Drive)](https://drive.google.com/drive/folders/1b6VibRKLjdxht6mp_nbSi8baZ2uqQ4Hf?usp=sharing)

After downloading, place them inside a `models/` folder at the repo root:
```
models
├── chest_CT_SCAN-DenseNet201.keras
├── chest_CT_SCAN-ResNet50.keras
├── chest_CT_SCAN-DenseNet201-tuned.keras
└── chest_CT_SCAN-ResNet50-tuned.keras
```


## Installation

Clone the repository:
```bash
git clone https://github.com/SShreyas17/lungCancerClassifier.git
cd lungCancerClassifier
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Running the GUI
```bash
python app.py
```
The GUI allows you to:

* Upload a CT scan image

* View classification probabilities from DenseNet201 and ResNet50

* See the final ensemble-based prediction



## Training and Evaluation
To retrain or verify the models:

* Use notebooks/model_training.ipynb for model building and training

* Use notebooks/model_evaluation.ipynb for performance evaluation



## Disclaimer

This project is for research and educational purposes only.
The predictions are not guaranteed to be accurate. Always consult a qualified medical professional for diagnosis and treatment.
