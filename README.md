# AI vs Human Text Classification App

This Streamlit app detects whether a given piece of text is **AI-generated** or **Human-written**, using trained machine learning models like Support Vector Machine (SVM), Decision Tree, and AdaBoost. The application also supports batch processing of text files (TXT, CSV, DOCX, PDF) and allows model comparison.

---

## Features

**Single Text Prediction**: Classify one input at a time.
**Batch Processing**: Upload a file and classify multiple texts at once.
**Model Comparison**: Compare predictions across all available models.
**Confidence Scores**: View prediction probabilities.
**Download Results**: Export batch results as CSV.

---

## Requirements

- Python 3.11
- pip
- Git

---

## Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/sonjaysri/ai_human_detection_project.git
cd ai_human_detection_project


2. python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # macOS/Linux

cd streamli_ml_app

3. **Install Dependencies**
pip install -r requirements.txt

4. streamlit run app.py
