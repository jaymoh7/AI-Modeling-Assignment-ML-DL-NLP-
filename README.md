# AI Modeling Assignment (ML, DL & NLP)

## Overview
This project showcases AI model building using classical ML (Scikit-learn), deep learning (PyTorch), and NLP (spaCy). It includes:

- Iris Species Classification
- Handwritten Digit Classification (MNIST)
- Named Entity Recognition & Sentiment Analysis
- Ethical Evaluation
- (Optional) Deployment via Streamlit

---

## Contents

| Section                        | File                             |
|--------------------------------|----------------------------------|
| Theoretical Questions          | report.pdf                       |
| Iris Classifier (Scikit-learn) | iris_classification.py           |
| MNIST CNN (PyTorch)            | mnist_cnn_pytorch.py             |
| spaCy NLP                      | spacy_ner_sentiment.py           |
| Streamlit App (Bonus)          | mnist_webapp.py                  |
| Model Screenshots              | [Add screenshots folder]         |

---

## Requirements

```bash
pip install torch torchvision scikit-learn spacy matplotlib streamlit
python -m spacy download en_core_web_sm

How to Run

python iris_classification.py
python mnist_cnn_pytorch.py
python spacy_ner_sentiment.py
streamlit run mnist_webapp.py

Ethical Considerations

Discussed in report output screenshot.pdf and includes reflection on model fairness and biases.
Authors

    James
