# Machine Learning Projects (`ml-projects-000a01bc`)

A comprehensive collection of 24 practical machine learning mini-projects implemented with Jupyter Notebooks and Python. This repository covers regression, classification, clustering, natural language processing (NLP), computer vision, speech processing, recommendation engines, and time series forecasting.

---

## 📺 Video Tutorials
Follow the step-by-step video playlist for these projects:
- **Complete Video Tutorial Playlist:** [Machine Learning Projects on YouTube](http://bit.ly/mlprojectsplaylist)

---

## 🚀 Quickstart & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Amit1092tiwari/ml-projects-000a01bc.git
cd ml-projects-000a01bc
```

### 2. Create and activate a virtual environment
```bash
# On Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# On Windows (Command Prompt / PowerShell)
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook or JupyterLab
```bash
jupyter lab
```

---

## 📂 Project Catalog

### 📈 Regression
| Project | Description | Key Algorithms & Libraries |
| :--- | :--- | :--- |
| [Bigmart Sales Prediction Analysis](./Bigmart%20Sales%20Prediction%20Analysis%20-%20Regression) | Predict item outlet sales based on store attributes and product features. | Ridge, Lasso, Linear Regression, Scikit-learn |
| [Bike Sharing Demand Analysis](./Bike%20Sharing%20Demand%20Analysis%20-%20Regression) | Hourly bike rental demand prediction combining weather and calendar features. | Random Forest, Extra Trees, Gradient Boosting |
| [Black Friday Sales Prediction Analysis](./Black%20Friday%20Sales%20Prediction%20Analysis%20-%20Regression) | Customer purchase amount prediction using demographic and product features. | Decision Trees, Random Forest, Scikit-learn |
| [Boston Housing Prediction Analysis](./Boston%20Housing%20Prediction%20Analysis%20-%20Regression) | House price prediction using neighborhood and housing characteristics. | Linear Regression, XGBoost, Scikit-learn |

### 🎯 Classification
| Project | Description | Key Algorithms & Libraries |
| :--- | :--- | :--- |
| [Breast Cancer Detection](./Breast%20Cancer%20Detection%20-%20Classification) | Diagnostic breast cancer classification using Wisconsin diagnostic dataset. | PyCaret, Random Forest, Logistic Regression |
| [Credit Card Fraud Detection](./Credit%20Card%20Fraud%20Detection%20-%20Classification) | Highly imbalanced transaction fraud detection with SMOTE and ensemble trees. | SMOTE, XGBoost, Random Forest, Scikit-learn |
| [Iris Dataset Analysis](./Iris%20dataset%20analysis%20-%20Classification) | Multi-class flower species classification from sepal and petal dimensions. | Logistic Regression, KNN, Decision Tree |
| [Loan Prediction Analysis](./Loan%20Prediction%20Analysis%20-%20Classification) | Predict customer loan eligibility and approval status based on profile data. | Random Forest, Extra Trees, Logistic Regression |
| [Titanic Dataset Analysis](./Titanic%20Dataset%20Analysis%20-%20Classification) | Passenger survival prediction on the Titanic using passenger demographics. | LightGBM, CatBoost, XGBoost, Scikit-learn |
| [Water Quality Classification](./Water%20Quality%20Classification) | Classify water potability based on chemical metrics and physical properties. | Decision Trees, Random Forest, Seaborn |
| [Wine Quality Prediction Analysis](./Wine%20Quality%20Prediction%20Analysis%20-%20Classification) | Quality score prediction for red and white wine variants. | Extra Trees, Random Forest, XGBoost |

### 🧩 Clustering & Unsupervised Learning
| Project | Description | Key Algorithms & Libraries |
| :--- | :--- | :--- |
| [Mall Customer Segmentation Analysis](./Mall%20Customer%20Segmenation%20Analysis%20-%20Clustering) | Customer segmentation based on annual income and spending score. | K-Means Clustering, Elbow Method |
| [Turkiye Student Evaluation Analysis](./Turkiye%20Student%20Evaluation%20Analysis%20-%20Clustering) | Student feedback and instructor evaluation clustering. | PCA, K-Means Clustering, Scikit-learn |

### 🔍 Recommendation Systems
| Project | Description | Key Algorithms & Libraries |
| :--- | :--- | :--- |
| [IMDB Movies Dataset Recommender](./IMDB%20Movies%20Dataset%20-%20Recommendation%20Engine) | Content-based movie recommendation engine based on metadata and plot similarity. | TF-IDF, Cosine Similarity, Scikit-learn |
| [Million Songs Dataset Recommender](./Million%20Songs%20Dataset%20-%20Recommendation%20Engine) | Collaborative song recommendation using popularity and item-similarity co-occurrence. | Jaccard Similarity, Item Co-occurrence Matrix |

### 💬 Natural Language Processing (NLP)
| Project | Description | Key Algorithms & Libraries |
| :--- | :--- | :--- |
| [SMS Spam Detection Analysis](./SMS%20Spam%20Detection%20Analysis%20-%20NLP) | Detect spam vs. ham messages using text pipelines and TF-IDF representation. | Scikit-learn Pipeline, Naive Bayes, SVC |
| [Twitter Sentiment Analysis](./Twitter%20Sentiment%20Analysis%20-%20NLP) | Sentiment classification and hashtag analysis of tweets with BoW and WordCloud. | NLTK, Stemming, CountVectorizer, WordCloud |
| [Realtime Twitter Sentiment Analysis](./Realtime%20Twitter%20Sentiment%20Analysis%20-%20NLP) | Live tweet streaming and sentiment inference using transformer embeddings. | Tweepy, Flair NLP, Pretrained Transformers |
| [Text Summarization using Word Frequency](./Text%20Summarization%20using%20Word%20Frequency%20-%20NLP) | Extractive text summarization based on normalized word frequencies and sentence scoring. | NLTK, Heapq, Frequency Scoring |

### 👁️ Computer Vision & OCR
| Project | Description | Key Algorithms & Libraries |
| :--- | :--- | :--- |
| [Face Detection (OpenCV)](./Face%20Detection%20(OpenCV)%20-%20Computer%20Vision) | Real-time face detection on images using Haar Cascade Classifiers. | OpenCV (`cv2`), Matplotlib |
| [Image to Text Conversion (OCR)](./Image%20to%20Text%20Conversion%20&%20Extraction%20-%20OCR) | Optical character recognition and structured data regex extraction from documents. | PyTesseract, Tesseract-OCR, Pillow, Regex |

### 🎙️ Speech Processing
| Project | Description | Key Algorithms & Libraries |
| :--- | :--- | :--- |
| [Speech to Text](./Speech%20to%20Text%20-%20Speech%20Recognition) | Convert live microphone input or `.wav` audio files into transcribed text. | SpeechRecognition, Google Speech API |
| [Text to Speech](./Text%20to%20Speech%20-%20Speech%20Synthesis) | Synthesize speech from text using Google TTS (`gTTS`) and offline `pyttsx3`. | gTTS, pyttsx3, IPython.display |

### ⏳ Time Series Forecasting
| Project | Description | Key Algorithms & Libraries |
| :--- | :--- | :--- |
| [Traffic Forecast](./Traffic%20Forecast%20-%20Time%20Series%20Analysis) | Hourly road traffic forecasting with trend and seasonality decomposition. | Facebook Prophet / Prophet |

---

## 🛠️ Repository Best Practices

- **Isolated Preprocessing Pipelines:** Always use `sklearn.pipeline.Pipeline` or fit scalers/encoders strictly on the training set to prevent data leakage.
- **Appropriate Evaluation Metrics:** In imbalanced classification (e.g., Fraud, Spam), prioritize Precision, Recall, and PR-AUC over raw accuracy.
- **Time Series Splitting:** For chronological datasets, use forward chaining or `TimeSeriesSplit` rather than standard random cross-validation.

---

## 🏗️ Enterprise MLOps & Production Architecture

This repository includes a production blueprint demonstrating how exploratory machine learning models are operationalized into **CI/CD Automation**, **Batch Processing**, and **Daily Scheduled Cadence**.

### 1. Architectural Topology
- **Continuous Integration (CI):** [`.github/workflows/ci.yml`](./.github/workflows/ci.yml) validates code style via `ruff`, runs the `pytest` test suite, and verifies that all 24 Jupyter notebooks parse without JSON corruption.
- **Daily Batch Orchestration:** [`.github/workflows/daily_batch_schedule.yml`](./.github/workflows/daily_batch_schedule.yml) triggers everyday at `02:00 AM UTC` to execute the daily traffic demand forecasting pipeline and export artifact metrics.
- **Airflow DAG:** [`dags/daily_traffic_batch_dag.py`](./dags/daily_traffic_batch_dag.py) provides a production-grade DAG definition for enterprise orchestrators (Apache Airflow / Cloud Composer).
- **Modular Pipeline Source:** [`src/traffic_forecast/`](./src/traffic_forecast/) contains modular data validation, feature engineering, model fitting with fallback mechanisms, and batch inference.
- **Containerization:** [`Dockerfile`](./Dockerfile) packages the batch forecasting environment into a hardened, non-root Linux container.

### 2. Running Batch Inference
```bash
python -m src.traffic_forecast.pipeline \
  --data-path "Traffic Forecast - Time Series Analysis/Traffic data.csv" \
  --output-dir "artifacts/traffic_forecast" \
  --forecast-days 30
```

### 3. Running the Test Suite
```bash
pytest tests/ -v
```

