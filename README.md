# QueueSense - Smart Queue Predictor

A beginner-friendly Machine Learning mini project. It is a small web application that **predicts the waiting time in a queue** from:

- number of people waiting
- number of active counters
- average service time per person

It also shows a **crowd status (Low / Medium / High)** and a **simple explanation** of the prediction.

**Tech used:** Python, Pandas, Scikit-learn (Linear Regression), Flask, HTML, CSS, JavaScript.
No deep learning, no APIs, no external AI services.

---

## 1. Project structure and what each file does

```
QueueSense/
├── app.py              # Flask web app: pages, input validation, prediction
├── train_model.py      # Reads dataset.csv, trains Linear Regression, saves model.pkl
├── dataset.csv         # Small sample dataset (40 rows)
├── model.pkl           # Trained model (created by train_model.py)
├── requirements.txt    # Libraries to install
├── .gitignore          # Files Git should ignore (venv, __pycache__)
├── templates/
│   ├── index.html      # Home page
│   └── predict.html    # Prediction page + result (with JavaScript validation)
└── static/
    └── style.css       # Colours, cards, responsive layout
```

| File | Job in one line |
|---|---|
| `train_model.py` | The "learning" part: teaches the model using `dataset.csv`. |
| `model.pkl` | The saved brain of the project, so we do not train again every time. |
| `app.py` | The "using" part: takes user input, asks the model, shows the answer. |
| `templates/*.html` | What the user sees in the browser. |
| `style.css` | Makes the pages look clean and work on mobile. |

---

## 2. How the Machine Learning works (explain this in your viva)

### Step 1 - The dataset
`dataset.csv` has 40 rows. Each row is one example of a queue:

| people_waiting | active_counters | avg_service_time | waiting_time |
|---|---|---|---|
| 57 | 3 | 3.0 | 61.8 |
| 39 | 2 | 2.0 | 42.7 |
| 42 | 5 | 6.0 | 54.5 |
| 55 | 1 | 5.0 | 262.8 |
| 36 | 5 | 1.5 | 13.1 |

- The first 3 columns are the **inputs (features)**.
- `waiting_time` (minutes) is the **output (label)** we want to predict.

> **Be honest in the viva:** this is a small *sample* dataset made for learning. It follows a realistic queue pattern with some random variation, because real queue data is hard to collect. A real project would use recorded data from a bank, hospital or canteen.

### Step 2 - Preparing the data (feature engineering)
Waiting time does not grow in a straight line with each column separately.
Logically:

```
total work        = people x service time
work per counter  = total work / counters   <- we call this "workload"
```

So `train_model.py` creates one new column, **workload = people x service time / counters**.
Waiting time is almost a straight line of workload, which is exactly what Linear Regression is good at.

Result when we ran both versions on the test data:

| Inputs used | R2 score |
|---|---|
| 3 raw columns | 0.754 |
| `workload` | 0.996 |

### Step 3 - Training and testing
- The data is split: **80% for training (32 rows)** and **20% for testing (8 rows)**.
- The model **learns only from the training rows**.
- Then we **test it on the 8 unseen rows** to check it works on new data, not only on data it has already seen.

### Step 4 - Linear Regression in simple words
Linear Regression finds the best straight line through the data points:

```
waiting_time = m x workload + c
```

- `m` (slope / coefficient): how many extra minutes of waiting for each extra minute of workload.
- `c` (intercept): the base waiting time when workload is zero.
- "Best line" means the line where the total of the squared differences between real and predicted values is smallest (the *least squares* method).

Our model learned: **waiting_time = 0.98 x workload + 2.64**

### Step 5 - Checking the quality
- **MAE (Mean Absolute Error):** average mistake in minutes. We got about **2.25 minutes**.
- **R2 score:** how well the line fits, from 0 to 1. We got **0.996**.

The score is very high because the sample data is clean and the relationship is simple. Real-world data would be noisier, so scores would be lower.

### Step 6 - Prediction
Example: 20 people, 2 counters, 3 minutes each.

```
workload = 20 x 3 / 2 = 30
waiting_time = 0.98 x 30 + 2.64 = about 32 minutes
```

### Crowd status (this part is a simple rule, not ML)
| Predicted waiting time | Status |
|---|---|
| up to 10 minutes | Low |
| 11 to 30 minutes | Medium |
| more than 30 minutes | High |

---

## 3. How to install and run in VS Code on Windows

### One-time setup
1. Install **Python 3.10 or newer** from python.org. On the first installer screen, **tick "Add python.exe to PATH"**.
2. Install **VS Code** and the **Python extension** (Extensions icon, search "Python").

### Run the project
1. Unzip the project. In VS Code: **File > Open Folder** and choose the `QueueSense` folder.
2. Open the terminal: **Terminal > New Terminal**.
3. Check Python: `python --version`
4. Create a virtual environment: `python -m venv venv`
5. Activate it: `venv\Scripts\activate`
   - If PowerShell shows a "running scripts is disabled" error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again (or open a **Command Prompt** terminal instead).
   - You should now see `(venv)` at the start of the line.
6. Install the libraries: `pip install -r requirements.txt`
7. Train the model: `python train_model.py`
   (You will see the dataset size, MAE, R2 and the learned formula. This also creates a fresh `model.pkl` that matches your scikit-learn version.)
8. Start the app: `python app.py`
9. Open **http://127.0.0.1:5000** in your browser. Press **Ctrl + C** in the terminal to stop.

### Common problems
| Problem | Fix |
|---|---|
| `python is not recognized` | Reinstall Python and tick "Add to PATH", then reopen VS Code. |
| `No module named flask` | Activate the venv (step 5) and run step 6 again. |
| `Could not load model.pkl` | Run `python train_model.py`. |
| Port 5000 already in use | In `app.py` change the last line to `app.run(debug=True, port=5001)`. |

---

## 4. How to test the project

Open the Predict page, enter the values and compare with the table.

| People | Counters | Service time | Expected result |
|---|---|---|---|
| 5 | 3 | 2 | about 6 min, **Low** |
| 12 | 2 | 3 | about 20 min, **Medium** |
| 20 | 2 | 3 | about 32 min, **High** |
| 45 | 3 | 4 | about 61 min, **High** |
| 0 | 1 | 3 | about 3 min, **Low** |

**Invalid input tests** (you should see a red error message, not a crash):

| Input | Message |
|---|---|
| People left empty | Number of people waiting is required. |
| People = -5 | Number of people waiting must be between 0 and 200. |
| People = 2.5 | Number of people waiting must be a whole number. |
| Counters = 0 | Number of active counters must be between 1 and 20. |
| Service time = 99 | Average service time must be between 0.5 and 30. |

Also check on mobile: press **F12** in the browser and click the phone icon to see the responsive layout.

Validation happens twice: JavaScript checks in the browser (fast message) and Python checks again on the server (safe even if JavaScript is off).

---

## 5. How to upload to GitHub

1. Install **Git** from git-scm.com and create a free account on github.com.
2. On GitHub click **New repository**. Name it `QueueSense`. Do **not** tick "Add a README" (you already have one). Click **Create repository**.
3. In the VS Code terminal (inside the project folder), run once:
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```
4. Then run:
```bash
git init
git add .
git commit -m "Initial commit: QueueSense Smart Queue Predictor"
git branch -M main
git remote add origin https://github.com/<your-username>/QueueSense.git
git push -u origin main
```
5. Refresh the GitHub page. Your code is now online. Add 2 screenshots to the README later.

---

## 6. Viva questions and short answers

**What does your project do?**
It predicts waiting time in a queue from people waiting, active counters and average service time, using Linear Regression.

**What is Machine Learning?**
Teaching a computer to find patterns from data and use them to predict new results, instead of writing fixed rules.

**Which type of ML is this?**
Supervised learning (we give inputs and correct answers) and regression (the answer is a number).

**What is Linear Regression?**
An algorithm that fits a straight line, `y = m x + c`, to the data and uses it to predict a number.

**Why did you create the `workload` column?**
Waiting time depends on people x service time divided by counters, which is not a straight line in the raw columns. Combining them first gave R2 0.996 instead of 0.754.

**Why split data into training and testing?**
To check the model on new data it has not seen, so we know it really learned and did not just memorise.

**What are MAE and R2?**
MAE is the average error in minutes (about 2.25). R2 shows how well the line fits (closer to 1 is better).

**Is your dataset real?**
No. It is a small sample dataset made for learning, following a realistic queue pattern with random variation.

**Why is the accuracy so high?**
Because the sample data is clean and the relationship is simple. Real data would give lower accuracy.

**What is `model.pkl`?**
The trained model saved to a file using `pickle`, so the app can load and use it without training again.

**What is Flask? What is a route?**
Flask is a small Python web framework. A route links a URL (like `/predict`) to a Python function.

**Is the Low / Medium / High status machine learning?**
No, it is a simple if-else rule on the predicted time.

**Limitations?**
Small dataset, assumes all counters work at the same speed, ignores people arriving later, and works best for values similar to the dataset (3 to 60 people, 1 to 5 counters, 1 to 6 minutes per person).

**Future improvements?**
Use real data, add features like time of day, try other models, save predictions in a database.

---

## 7. Limitations
- Sample data, not real-world data.
- Assumes constant service time and equal speed at all counters.
- Predictions outside the dataset range are extrapolations and less reliable.
