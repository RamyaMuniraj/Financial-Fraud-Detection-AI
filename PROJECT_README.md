\# Financial Fraud Detection AI



\## Project Overview

A machine learning project that analyzes financial transactions and predicts whether they are legitimate or potentially fraudulent. The project prepares a saved model for future Streamlit and Power BI dashboard integration.



\## Technologies

\- Python

\- Pandas and NumPy

\- Scikit-learn

\- Matplotlib and Seaborn

\- Streamlit

\- Power BI

\- Joblib



\## Dataset

File: `data/synthetic\_fraud\_dataset\_v2.csv`



\- Total transactions: 50,000

\- Fraudulent transactions: 3,963

\- Legitimate transactions: 46,037

\- Target column: `Fraud\_Label`

\- 0 = Legitimate

\- 1 = Fraudulent



The dataset is synthetic and does not represent guaranteed real-world performance.



\## Model

Saved model: `models/fraud\_detection\_model.joblib`



The selected model is a Logistic Regression pipeline with a classification threshold selected using validation data.



\## Test Results



| Metric | Result |

|---|---:|

| Accuracy | 88.94% |

| Precision | 34.61% |

| Recall | 44.57% |

| F1-score | 38.96% |

| ROC-AUC | 0.7437 |

| PR-AUC | 0.2658 |



The reserved test set contains 10,000 transactions, including 792 fraudulent transactions.



\## Evaluation Outputs

\- `outputs/final\_test\_metrics.csv`

\- `outputs/confusion\_matrix.csv`

\- `outputs/classification\_report.txt`

\- `outputs/model\_comparison.csv`

\- `outputs/threshold\_comparison.csv`



\## Run the Evaluation



Activate the virtual environment and run:



```powershell

python evaluate\_saved\_model.py

```



\## Future Enhancements

\- Improve fraud detection and reduce missed fraud cases.

\- Develop an interactive Streamlit dashboard.

\- Create a Power BI dashboard for fraud analysis.

\- Evaluate performance on independent, realistic transaction data.



\## Disclaimer

This project is for educational purposes and uses synthetic data. It is not a production-ready banking fraud detection system.



