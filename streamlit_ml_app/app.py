# STREAMLIT ML CLASSIFICATION APP - DUAL MODEL SUPPORT
# =====================================================

import re
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import unicodedata 
import docx
import fitz
import nltk

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize


for resource in [
    'punkt',
    'punkt_tab',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng'
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)




#  Advanced Text Preprocessing with Lemmatization

class TextPreprocessor:
    """
    Advanced text preprocessing class with lemmatization
    This will clean text better than basic preprocessing
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Add custom stop words
        self.stop_words.update(['u', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure'])

    def get_wordnet_pos(self, word):
        """Convert POS tag to format accepted by WordNet lemmatizer"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def clean_text(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs, emails, HTML tags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'<.*?>', '', text)

        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        
        text = ' '.join(text.split())

        return text

    def lemmatize_text(self, text):
        """Apply lemmatization with POS tagging"""
        tokens = word_tokenize(text)

        lemmatized_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                # Get POS tag and lemmatize
                pos_tag = self.get_wordnet_pos(token)
                lemmatized_token = self.lemmatizer.lemmatize(token, pos_tag)
                lemmatized_tokens.append(lemmatized_token)

        return ' '.join(lemmatized_tokens)

    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        cleaned_text = self.clean_text(text)
        lemmatized_text = self.lemmatize_text(cleaned_text)
        return lemmatized_text


# Page Configuration
st.set_page_config(
    page_title="AI vs Human Text Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FAEBD7;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FAEBD7;
        border: 1px solid #FAEBD7;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #FAEBD7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING SECTION
# ============================================================================

@st.cache_resource
def load_models():
    models = {}
    # base_path = '/content/drive/MyDrive/Models/' # Removed base_path

    try:
        # Load TF-IDF vectorizer
        try:
            models['vectorizer'] = joblib.load('models/tfidf_vectorizer.pkl')
            models['vectorizer_available'] = True
        except FileNotFoundError:
            models['vectorizer_available'] = False
            st.error(f"Vectorizer file not found: tfidf_vectorizer.pkl")


        # Load SVM model
        try:
            models['svm'] = joblib.load('models/svm_model.pkl')
            models['svm_available'] = True
        except FileNotFoundError:
            models['svm_available'] = False
            st.warning(f"SVM model file not found: svm_model.pkl")


        # Load Decision Tree model
        try:
            models['decision_tree'] = joblib.load('models/dt_model.pkl')
            models['dt_available'] = True
        except FileNotFoundError:
            models['dt_available'] = False
            st.warning(f"Decision Tree model file not found: dt_model.pkl")


        # Load AdaBoost model
        try:
            models['adaboost'] = joblib.load('models/ab_model.pkl')
            models['ab_available'] = True
        except FileNotFoundError:
            models['ab_available'] = False
            st.warning(f"AdaBoost model file not found: ab_model.pkl")


        # Check if at least one complete setup is available (vectorizer and at least one classifier)
        any_classifier_available = models.get('svm_available', False) or models.get('dt_available', False) or models.get('ab_available', False)

        if not (models.get('vectorizer_available', False) and any_classifier_available):
            st.error("No complete model setup found! Ensure vectorizer and at least one classifier are available.")
            return None

        return models

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def make_prediction(text, model_choice, models):
    """Make prediction using the selected model"""
    if models is None:
        return None, None

    try:
        prediction = None
        probabilities = None

        # Preprocess the text using the spacy_preprocess function
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.preprocess(text)

        if not processed_text and text.strip(): # Handle case where preprocessing fails but input wasn't empty
             st.warning("Text preprocessing failed. Cannot make prediction.")
             return None, None


        # Assuming the loaded models are pipelines that include the vectorizer
        # and were trained on processed text.
        # If your loaded models are just the classifiers, you would need to:
        # 1. Vectorize the processed_text using models['vectorizer'].transform([processed_text])
        # 2. Pass the vectorized text to the loaded classifier models.

        # Since the notebook saved the best estimators from GridSearchCV (which are pipelines),
        # we can directly use the loaded pipelines with the raw text, and the pipeline
        # will handle the preprocessing and vectorization steps internally as they were
        # defined in the original pipeline structure.

        if model_choice == "svm" and models.get('svm_available'):
            # Use the preprocessed text for prediction
            prediction = models['svm'].predict([processed_text])[0]
            # SVM predict_proba might not be calibrated or available for linear kernel
            try:
                 # Use the preprocessed text for predict_proba
                 probabilities = models['svm'].predict_proba([processed_text])[0]
            except AttributeError:
                 st.warning("Probability prediction not available for selected SVM configuration.")
                 probabilities = None

        elif model_choice == "decision_tree" and models.get('dt_available'):
            # Use the preprocessed text for prediction
            prediction = models['decision_tree'].predict([processed_text])[0]
            # Use the preprocessed text for predict_proba
            probabilities = models['decision_tree'].predict_proba([processed_text])[0]

        elif model_choice == "adaboost" and models.get('ab_available'):
            # Use the preprocessed text for prediction
            prediction = models['adaboost'].predict([processed_text])[0]
            # Use the preprocessed text for predict_proba
            probabilities = models['adaboost'].predict_proba([processed_text])[0]

        if prediction is not None:
            # Convert to readable format
            # Assuming 0 for Human, 1 for AI based on your training data
            class_names = ['Human', 'AI']
            prediction_label = class_names[prediction]
            return prediction_label, probabilities
        else:
            return None, None

    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error(f"Model choice: {model_choice}")
        st.error(f"Available models: {[k for k, v in models.items() if isinstance(v, bool) and v]}")
        return None, None

def get_available_models(models):
    """Get list of available models for selection"""
    available = []

    if models is None:
        return available

    # Check if vectorizer is available as it's needed for individual classifiers
    vectorizer_available = models.get('vectorizer_available', False)

    if models.get('svm_available'):
        available.append(("svm", "📈 Support Vector Machine"))

    if models.get('dt_available'):
        available.append(("decision_tree", "🌳 Decision Tree"))

    if models.get('ab_available'):
        available.append(("adaboost", "🚀 AdaBoost"))

    return available

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Choose what you want to do:")

page = st.sidebar.selectbox(
    "Select Page:",
    ["🏠 Home", "🔮 Single Prediction", "📁 Batch Processing", "⚖️ Model Comparison", "📊 Model Info", "❓ Help"]
)

# Load models
models = load_models()

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🤖 AI vs Human Text Classification App</h1>', unsafe_allow_html=True)

    st.markdown("""
    Welcome to your machine learning web application! This app demonstrates classification
    of text as either **Human-written** or **AI-generated** using multiple trained models:
    **Support Vector Machine (SVM)**, **Decision Tree**, and **AdaBoost**.
    """)

    # App overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔮 Single Prediction
        - Enter text manually
        - Choose between models
        - Get instant predictions
        - See confidence scores (where available)
        """)

    with col2:
        st.markdown("""
        ### 📁 Batch Processing
        - Upload text files
        - Process multiple texts
        - Compare model performance
        - Download results
        """)

    with col3:
        st.markdown("""
        ### ⚖️ Model Comparison
        - Compare different models
        - Side-by-side results
        - Agreement analysis
        - Performance metrics
        """)

    # Model status
    st.subheader("📋 Model Status")
    if models:
        st.success("✅ Models loaded successfully!")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if models.get('svm_available'):
                st.info("**📈 SVM**\n✅ Available")
            else:
                st.warning("**📈 SVM**\n❌ Not Available")

        with col2:
            if models.get('dt_available'):
                st.info("**🌳 Decision Tree**\n✅ Available")
            else:
                st.warning("**🌳 Decision Tree**\n❌ Not Available")

        with col3:
            if models.get('ab_available'):
                st.info("**🚀 AdaBoost**\n✅ Available")
            else:
                st.warning("**🚀 AdaBoost**\n❌ Not Available")

        with col4:
            if models.get('vectorizer_available'):
                st.info("**🔤 TF-IDF Vectorizer**\n✅ Available")
            else:
                st.warning("**🔤 TF-IDF Vectorizer**\n❌ Not Available")

    else:
        st.error("❌ Models not loaded. Please check model files.")

# ============================================================================
# SINGLE PREDICTION PAGE
# ============================================================================

elif page == "🔮 Single Prediction":
    st.header("🔮 Make a Single Prediction")
    st.markdown("Enter text below and select a model to get AI vs Human text detection predictions.")

    if models:
        available_models = get_available_models(models)

        if available_models:
            # Model selection
            model_choice = st.selectbox(
                "Choose a model:",
                options=[model[0] for model in available_models],
                format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
            )

            # Text input
            user_input = st.text_area(
                "Enter your text here:",
                placeholder="Type or paste text to detect if it's AI or Human written...",
                height=150
            )

            # Character count
            if user_input:
                st.caption(f"Character count: {len(user_input)} | Word count: {len(user_input.split())}")

            # Example texts
            with st.expander("📝 Try these example texts"):
                # Replace with examples relevant to AI vs Human detection
                examples = [
                    "The quick brown fox jumps over the lazy dog.", # Likely Human
                    "As an AI language model, I can generate text on various topics.", # Likely AI
                    "This essay discusses the impact of climate change on global ecosystems.", # Could be either, depends on style
                    "I woke up this morning, made coffee, and read the newspaper.", # Likely Human
                    "Generating creative content is a core capability of modern artificial intelligence systems." # Likely AI
                ]

                col1, col2 = st.columns(2)
                for i, example in enumerate(examples):
                    with col1 if i % 2 == 0 else col2:
                        if st.button(f"Example {i+1}", key=f"example_{i}"):
                            st.session_state.user_input = example
                            st.rerun()

            # Use session state for user input
            if 'user_input' in st.session_state:
                user_input = st.session_state.user_input

            # Prediction button
            if st.button("🚀 Predict", type="primary"):
                if user_input.strip():
                    with st.spinner('Analyzing text...'):
                        prediction, probabilities = make_prediction(user_input, model_choice, models)

                        if prediction: # Check if prediction was successful
                            # Display prediction
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                if prediction == "Human":
                                    st.success(f"🎯 Prediction: **{prediction} Written**")
                                else:
                                    st.error(f"🎯 Prediction: **{prediction} Written**")

                            if probabilities is not None: # Display confidence only if available
                                with col2:
                                    confidence = max(probabilities)
                                    st.metric("Confidence", f"{confidence:.1%}")

                                # Create probability chart
                                st.subheader("📊 Prediction Probabilities")

                                # Detailed probabilities
                                col1, col2 = st.columns(2)
                                class_names = ['Human', 'AI'] # Ensure correct order
                                with col1:
                                    st.metric("🧍 Human", f"{probabilities[0]:.1%}")
                                with col2:
                                    st.metric("🤖 AI", f"{probabilities[1]:.1%}")

                                # Bar chart
                                prob_df = pd.DataFrame({
                                    'Category': class_names,
                                    'Probability': probabilities
                                })
                                st.bar_chart(prob_df.set_index('Category'), height=300)

                        else:
                            st.error("Failed to make prediction")
                else:
                    st.warning("Please enter some text to classify!")
        else:
            st.error("No models available for prediction.")
    else:
        st.warning("Models not loaded. Please check the model files.")

# ============================================================================
# BATCH PROCESSING PAGE
# ============================================================================

elif page == "📁 Batch Processing":
    st.header("📁 Upload File for Batch Processing")
    st.markdown("Upload a text, Word, PDF, or CSV file to process multiple texts for AI vs Human detection.")

    if models:
        available_models = get_available_models(models)

        if available_models:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'csv', 'docx', 'pdf'],
                help="Upload a .txt, .csv, .docx, or .pdf file"
            )

            if uploaded_file:
                model_choice = st.selectbox(
                    "Choose model for batch processing:",
                    options=[model[0] for model in available_models],
                    format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
                )

                if st.button("📊 Process File"):
                    try:
                        texts = []

                        if uploaded_file.type == "text/plain":
                            content = str(uploaded_file.read(), "utf-8")
                            texts = [line.strip() for line in content.split('\n') if line.strip()]

                        elif uploaded_file.type == "text/csv":
                            df = pd.read_csv(uploaded_file)
                            texts = df.iloc[:, 0].astype(str).dropna().tolist()

                        elif uploaded_file.name.endswith(".docx"):
                            doc = docx.Document(uploaded_file)
                            texts = [para.text.strip() for para in doc.paragraphs if para.text.strip()]

                        elif uploaded_file.name.endswith(".pdf"):
                            pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                            for page in pdf_doc:
                                text = page.get_text().strip()
                                if text:
                                    texts.extend([line.strip() for line in text.split('\n') if line.strip()])
                            pdf_doc.close()

                        if not texts:
                            st.error("No valid text found in the file.")
                        else:
                            st.info(f"Processing {len(texts)} texts...")
                            results = []
                            progress_bar = st.progress(0)

                            for i, text in enumerate(texts):
                                if text.strip():
                                    prediction, probabilities = make_prediction(text, model_choice, models)
                                    result_entry = {
                                        'Text': text[:100] + "..." if len(text) > 100 else text,
                                        'Full_Text': text,
                                        'Prediction': prediction if prediction is not None else "Error",
                                        'Confidence': f"{max(probabilities):.1%}" if probabilities is not None else "N/A",
                                        'Human_Prob': f"{probabilities[0]:.1%}" if probabilities is not None else "N/A",
                                        'AI_Prob': f"{probabilities[1]:.1%}" if probabilities is not None else "N/A"
                                    }
                                    results.append(result_entry)
                                progress_bar.progress((i + 1) / len(texts))


                            st.success(f"✅ Processed {len(results)} texts successfully!")
                            results_df = pd.DataFrame(results)

                            st.subheader("📊 Summary Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            human_count = sum(1 for r in results if r['Prediction'] == 'Human')
                            ai_count = sum(1 for r in results if r['Prediction'] == 'AI')
                            valid_confidences = [float(r['Confidence'].strip('%')) for r in results if r['Confidence'] != "N/A"]
                            avg_confidence = np.mean(valid_confidences) if valid_confidences else 0

                            with col1:
                                st.metric("Total Processed", len(results))
                            with col2:
                                st.metric("🧍 Human", human_count)
                            with col3:
                                st.metric("🤖 AI", ai_count)
                            with col4:
                                st.metric("Avg Confidence", f"{avg_confidence:.1f}%" if avg_confidence else "N/A")

                            st.subheader("📋 Results Preview")
                            st.dataframe(results_df[['Text', 'Prediction', 'Confidence']], use_container_width=True)

                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Full Results",
                                data=csv,
                                file_name=f"predictions_{model_choice}_{uploaded_file.name}.csv",
                                mime="text/csv"
                            )

                    except Exception as e:
                        st.error(f"Error processing file: {e}")
            else:
                st.info("Please upload a file to get started.")
                with st.expander("📄 Example File Formats"):
                    st.markdown("""
                    - **Text File (.txt):** One entry per line  
                    - **CSV (.csv):** Text in the first column  
                    - **Word (.docx):** Paragraphs as separate entries  
                    - **PDF (.pdf):** Extracted per line  
                    """)
        else:
            st.error("No models available for batch processing.")
    else:
        st.warning("Models not loaded. Please check the model files.")



# ============================================================================
# MODEL COMPARISON PAGE
# ============================================================================

elif page == "⚖️ Model Comparison":
    st.header("⚖️ Compare Models")
    st.markdown("Compare predictions from different models on the same text.")

    if models:
        available_models = get_available_models(models)

        if len(available_models) >= 2:
            comparison_text = st.text_area(
                "Enter text to compare models:",
                placeholder="Paste or type a passage to see how each model evaluates it...",
                height=120
            )

            if st.button("📊 Compare All Models") and comparison_text.strip():
                st.subheader("🔍 Model Comparison Results")

                comparison_results = []
                class_names = ['Human', 'AI']

                for model_key, model_name in available_models:
                    pred_label, probs = make_prediction(comparison_text, model_key, models)
                    if pred_label and probs is not None:
                        comparison_results.append({
                            "Model": model_name,
                            "Prediction": pred_label,
                            "Confidence": f"{max(probs) * 100:.1f}%",
                            "Human %": f"{probs[0] * 100:.1f}%",
                            "AI %": f"{probs[1] * 100:.1f}%",
                            "Raw_Probs": probs
                        })

                if comparison_results:
                    df = pd.DataFrame(comparison_results)
                    st.table(df[["Model", "Prediction", "Confidence", "Human %", "AI %"]])

                    # Agreement Check
                    predictions = [r["Prediction"] for r in comparison_results]
                    if len(set(predictions)) == 1:
                        st.success(f"✅ All models agree: **{predictions[0]}**")
                    else:
                        st.warning("⚠️ Models disagree:")
                        for r in comparison_results:
                            st.write(f"- {r['Model']}: **{r['Prediction']}**")

                    # Side-by-side bar charts
                    st.subheader("📊 Probability Breakdown")
                    cols = st.columns(len(comparison_results))

                    for i, result in enumerate(comparison_results):
                        with cols[i]:
                            st.markdown(f"**{result['Model']}**")
                            chart_df = pd.DataFrame({
                                "Category": class_names,
                                "Probability": result["Raw_Probs"]
                            })
                            st.bar_chart(chart_df.set_index("Category"))
                else:
                    st.error("Could not retrieve valid predictions.")
        else:
            st.info("At least two models are required for comparison.")
    else:
        st.error("❌ Models not loaded.")


# ============================================================================
# MODEL INFO PAGE
# ============================================================================

elif page == "📊 Model Info":
    st.header("📊 Model Information")

    if models:
        st.success("✅ Models are loaded and ready!")

        st.subheader("🔧 Available Models")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📈 Support Vector Machine (SVM)
            **Type:** Margin-based Classifier  
            **Kernel:** Linear (or RBF depending on training)  
            **Use Case:** Text classification with high-dimensional data  
            **Strengths:**
            - High accuracy
            - Works well with TF-IDF vectors
            - Handles sparse data effectively
            """)

        with col2:
            st.markdown("""
            ### 🌳 Decision Tree
            **Type:** Rule-based Tree Classifier  
            **Criterion:** Gini or Entropy  
            **Use Case:** Interpretable AI detection  
            **Strengths:**
            - Easy to visualize
            - Fast inference
            - Handles non-linear patterns
            """)

        st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("""
            ### 🚀 AdaBoost
            **Type:** Ensemble of Weak Learners (usually Decision Trees)  
            **Use Case:** Boost performance of shallow models  
            **Strengths:**
            - Improved accuracy via boosting
            - Combines multiple models
            - Robust to overfitting with tuning
            """)

        with col4:
            st.markdown("""
            ### 🔤 TF-IDF Vectorizer
            **Method:** Term Frequency–Inverse Document Frequency  
            **Configuration:**
            - **Max Features:** 5,000
            - **N-grams:** (1,2) → unigrams + bigrams
            - **Min Document Frequency:** 2
            - **Stop Words:** English (filtered)
            """)

        st.markdown("---")
        st.subheader("📁 Model Files Status")

        files_to_check = [
            ("tfidf_vectorizer.pkl", "TF-IDF Vectorizer", models.get('vectorizer_available', False)),
            ("svm_model.pkl", "SVM Classifier", models.get('svm_available', False)),
            ("decision_tree_model.pkl", "Decision Tree Classifier", models.get('dt_available', False)),
            ("adaboost_model.pkl", "AdaBoost Classifier", models.get('ab_available', False))
        ]

        file_status = []
        for filename, description, status in files_to_check:
            file_status.append({
                "File": filename,
                "Description": description,
                "Status": "✅ Loaded" if status else "❌ Missing"
            })

        st.table(pd.DataFrame(file_status))

        st.subheader("🧠 Training Information")
        st.markdown("""
        - **Dataset:** AI-generated vs Human-written text corpus  
        - **Labels:** 0 = Human, 1 = AI  
        - **Preprocessing:** NLTK-based lemmatization, stopword removal, accent normalization
        - **Feature Extraction:** TF-IDF (1,2)-grams  
        - **Model Selection:** GridSearchCV with `f1_macro` scoring  
        - **Deployment:** Streamlit web app
        """)
    else:
        st.warning("❌ Models not loaded. Please check the files in the `models/` directory.")


# ============================================================================
# HELP PAGE
# ============================================================================

elif page == "❓ Help":
    st.header("❓ How to Use This App")

    with st.expander("🔮 Single Prediction"):
        st.write("""
        1. **Select a model** from the dropdown (SVM, Decision Tree, or AdaBoost)
        2. **Paste or type a piece of text** to be analyzed
        3. **Click 'Predict'** to classify the text as either AI-generated or Human-written
        4. **View the result:** see the classification, confidence score, and probability breakdown
        5. **Try the example texts** for quick testing
        """)

    with st.expander("📁 Batch Processing"):
        st.write("""
        1. **Prepare your file**:
           - **.txt file:** One document per line
           - **.csv file:** Ensure text is in the first column
        2. **Upload the file** using the uploader
        3. **Select a model** for processing
        4. **Click 'Process File'** to analyze all texts
        5. **Download a CSV** with predictions and probabilities
        """)

    with st.expander("⚖️ Model Comparison"):
        st.write("""
        1. **Enter any text** to test how different models compare
        2. **Click 'Compare All Models'**
        3. **View comparison table**: predictions, confidence, and probabilities
        4. **Check agreement**: see if models agree or differ
        5. **Visualize probabilities** via side-by-side bar charts
        """)

    with st.expander("🔧 Troubleshooting"):
        st.write("""
        **Common Issues and Solutions:**

        **❌ Models not loading**
        - Ensure model `.pkl` files exist in the `models/` directory:
          - `tfidf_vectorizer.pkl` (required)
          - `svm_model.pkl`
          - `decision_tree_model.pkl`
          - `adaboost_model.pkl`

        **❌ Prediction errors**
        - Make sure input text is not empty or too short
        - Avoid uploading corrupt or unsupported files
        - If using CSV, ensure text is in the first column

        **❌ File upload issues**
        - Accepted formats: `.txt` or `.csv`
        - File should be UTF-8 encoded
        - Remove any empty rows or corrupted data
        """)

    st.subheader("💻 Your Project Structure")
    st.code("""
    ai_human_detection_project/
    ├── app.py                          # Streamlit application
    ├── requirements.txt               # Python dependencies
    ├── models/                        # Saved ML models
    │   ├── svm_model.pkl
    │   ├── decision_tree_model.pkl
    │   ├── adaboost_model.pkl
    │   └── tfidf_vectorizer.pkl
    └── data/
        ├── training_data/
        └── test_data/
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 App Information")
st.sidebar.info("""
**AI vs Human Text Detection App**
Built with Streamlit

**Models:**  
- 📈 Support Vector Machine (SVM)  
- 🌳 Decision Tree  
- 🚀 AdaBoost  

**Vectorizer:** TF-IDF (1-2 grams)  
**Deployment:** Streamlit Cloud Ready  
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    Built with ❤️ using Streamlit  
    <br>
    <small>AI vs Human Detection | Machine Learning Project</small><br>
    <small>Powered by scikit-learn & NLTK</small>

</div>
""", unsafe_allow_html=True)
