# Loan Approval Prediction

A machine learning web application that predicts whether a loan application is likely to be approved based on applicant and loan information.

## 🚀 Features

* Interactive loan application form
* Data preprocessing and missing-value handling
* One-hot encoding of categorical features
* Feature scaling using `MinMaxScaler`
* Logistic Regression-based prediction
* Loan approval probability
* Flask backend
* Simple responsive frontend

## 🛠️ Technologies Used

* **Python**
* **Flask**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **HTML**
* **CSS**
* **JavaScript**

## 📂 Project Structure

```text
Loan_Processor/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── data/
│   └── train_loan.csv
│
├── model/
│   └── loan_model.pkl
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── script.js
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Loan_Processor.git
cd Loan_Processor
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## 🧠 Train the Model

Run:

```bash
python train_model.py
```

This preprocesses the dataset, trains the Logistic Regression model, evaluates it, and saves the trained model.

## ▶️ Run the Application

Start the Flask server:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Enter the applicant details and click **Predict Loan Approval**.

## 🔄 Workflow

```text
User Input
    ↓
Flask Backend
    ↓
Data Preprocessing
    ↓
Trained ML Model
    ↓
Prediction
    ↓
Approval Probability
    ↓
Frontend Result
```

## 📊 Model

The project uses **Logistic Regression** for loan approval classification. The input data is preprocessed using missing-value handling, one-hot encoding, and Min-Max scaling before prediction.

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes**. The predictions should not be used as a real-world financial or lending decision.

---

### 👨‍💻 Author

**Mitash Sharma**

BCA AI & Data Science | Machine Learning & Data Analysis
