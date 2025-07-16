<div align="center">

# 🤖 Affective AI — Understanding Emotions through Your Voice

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenAI](https://img.shields.io/badge/OpenAI-Whisper-blue.svg)](https://openai.com/research/whisper)
[![XGBoost](https://img.shields.io/badge/XGBoost-Integrated-orange.svg)](https://xgboost.ai/)

*Empowering machines to understand human emotions through Audio data.*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Tech Stack](#-tech-stack) • [License](#-license) • [Contact](#-contact)

</div>

---

## 🌟 Overview

**Affective AI** is a cutting-edge application designed to detect and analyze emotions from textual inputs. Leveraging advanced natural language processing and machine learning techniques, it provides insights into the emotional undertones present in text, aiding in areas like sentiment analysis, customer feedback interpretation, and more.

---

## 📌 Features

- ✅ **Emotion Detection**: Analyze text to identify underlying emotions.
- ✅ **Real-time Analysis**: Immediate feedback on emotional content.
- ✅ **User-friendly Interface**: Intuitive design for seamless interaction.
- ✅ **Data Visualization**: Graphical representation of emotion metrics.
- ✅ **Session Tracking**: Monitor and log user interactions for insights.

---

## 🛠️ Installation

#### Repository Cloning

```bash
# Clone the repository
git clone https://github.com/priyam-hub/Affective-AI.git

# Navigate into the directory
cd Affective-AI
```

#### Environmental Setup and Dependency Installation

```bash
# Environmental Setup and Dependency Installation
bash setup.sh
```
---

## 🚀 Usage

```bash
# Run the Streamlit application
streamlit run app.py
```

Upon running, navigate to the provided local URL in your browser to interact with the Affective AI application.

---

## 🧠 Tech Stack

### Core Technologies
- **Python 3.10+** -  Primary programming language.
- **OpenAI Whisper** -  Advanced speech recognition model.
- **Pandas & NumPy** -  Data manipulation and numerical operations.
- **XGBoost** -  Optimized gradient boosting library for classification tasks.
- **Plotly & Altair** -  Interactive data visualization libraries.
- **Streamlit** - Framework for building interactive web applications.

### Models Stack
- **Audio Generation**: OpenAI Whisper-Base
- **Emotion Generation**: Multinomial Logistic Regression, Multinomial Naive Bayes, Random Forest

### 🔊 Whisper Base Model (Fine-tuned)

This project uses a fine-tuned version of OpenAI's Whisper base model for robust and efficient speech-to-text transcription. It has been trained to handle conversational audio with improved accuracy.

👉 [View on Hugging Face](https://huggingface.co/priyammmmm/whisper_base)

---


## 📋 Prerequisites

Before you begin using the AI-Powered Emotion Detection System, ensure that your environment is properly set up. You will need to install the following tools and libraries:

### System and Software Requirements:
- **Operating System**: Linux, macOS, or Windows (All platforms supported)
- **Python** (3.10+): This project is built with Python 3.10 or newer. You can download Python from the official website:
   - [Python Download](https://www.python.org/downloads/)

---

## 🗺️ Roadmap

<details>
  <summary><strong>📍 Phase 1: Foundational Setup <em>(0 - 2 months)</em></strong></summary>

* 🎙️ **Audio Input & STT Integration**

  * Integrate OpenAI Whisper for speech-to-text conversion.
* 🧹 **Text Preprocessing**

  * Clean and structure text data from transcribed audio.
* 💬 **Basic Emotion Detection**

  * Implement baseline models:

    * Multinomial Naive Bayes
    * Logistic Regression
* 🖥️ **Streamlit Integration**

  * Build initial user interface for interaction.

</details>

<details>
  <summary><strong>🚀 Phase 2: Model Enhancement & Visualization <em>(2 - 4 months)</em></strong></summary>

* 🤖 **Advanced Emotion Detection Models**

  * Integrate:

    * Random Forest (with XGBoost)
    * Multilayer Perceptron
* 📊 **Real-Time Feedback & Visualization**

  * Use Plotly/Altair for interactive emotion graphs.
* 🗂️ **User Session Tracking**

  * Log user inputs and detected emotions in a local SQLite database.

</details>

<details>
  <summary><strong>🌐 Phase 3: Expansion & Deployment <em>(4 - 6 months)</em></strong></summary>

* 🌍 **Multi-language Emotion Detection**

  * Expand to multilingual transcription and analysis.
* 🧠 **Personalized Feedback Engine**

  * Analyze historical data for tailored emotion insights.
* 📦 **Containerization & Cloud Deployment**

  * Dockerize app
  * Deploy using AWS/GCP/Azure

</details>

---

## 📁 Project Structure

```plaintext
Affective-AI/
├── .gitignore                                # Ignoring files for Git
├── app.py                                    # Streamlit App
├── Dockerfile                                # Stored the Docker Setup
├── LICENSE                                   # MIT License
├── main.py                                   # Run the Machine Learning Pipeline
├── README.md                                 # Project documentation
├── requirements.txt                          # Python dependencies
├── setup.sh                                  # Package installation configuration
├── test.py                                   # Run the Machine Learning Pipeline
├── .devcontainer/                            # Directory to store the containers for Docker
│   └── devcontainer.json                     # Docker Container configuration
├── .streamlit/                               # Directory to store configuration of Streamlit
│   └── config.toml                           # Streamlit Configuration
├── .vscode/                                  # Directory to store configuration of VS Code
│   └── settings.json                         # VS Code configuration
├── audio/                                    # Directory to store the recorded audio file
│   └── recorded_audio.wav                    # Recorded audio stored in .wav format
├── config/                                   # Configuration files
│   ├── __init__.py                           
│   └── config.py/                            # All Configuration Variables of Pipeline
├── docs/                                     # Documents Directory
├── data/                                     # Data directory
│   ├── emotion_dataset_cleaned.csv           # Cleaned Emotion Dataset
│   └── emotion_dataset_raw.csv               # Raw Emotion Dataset
├── database/                                 # Directory to store the behaviour of the Client
│   └── data.db                               # Database file to store the behaviour of the user
├── images/                                   # Image Directory
│   ├── home_banner.jpg                       # Image for Home page
│   └── sidebar_logo.jpg                      # Logo for Sidebar
├── models/                                   # Directory to store the Models
│   ├── grammarly/                            # Directory to store the Grammarly Model
|   ├── whisper/                              # Directory to store the Base Model of OpenAI-Whisper
│   └── classification_models/                # Directory to store the ML Classification Models
│       ├── linear_svc.pkl                         # Pickle file for Linear Support Vector Classifier
│       ├── mlp_classifier.pkl                     # Pickle file for Multilayer Perceptron
│       ├── multinomial_logistic_regression.pkl    # Pickle file for Multinomial Logistic Regression
│       ├── multinomial_naive_bayes.pkl            # Pickle file for Multinomial Naive Bayes
│       ├── random_forest.pkl                      # Pickle file for Random Forest with XGBoost
│       └── rbf_svm.pkl                            # Pickle file for Support Vector Machine with RBF Kernel
├── notebooks/                                # Jupyter notebooks for experimentation
│   └── emotion_detection.ipynb               # Experimented Emotion Detection in Jupyter Notebook
├── results/                                  # Reports of the Project
│   ├── eda_results/                          # Directory to store the results of EDA
│   └── model_accuracy_comparison.png         # Result of different ML Models Accuracy
├── src/                                      # Source code
│   ├── audio_recorder/                       # Audio Recorder Directory
│   │   ├── __init__.py  
│   │   └── record_audio.py                   # Python file to record Audio                                           
│   ├── audio_transcriber/                    # Audio Transcriber Directory
│   │   ├── __init__.py   
│   │   └── transcribe_audio.py               # Python file to transcribe Audio                                      
│   ├── data_cleaner/                         # Data Cleaner Directory
│   │   ├── __init__.py                           
│   │   └── cleaner.py                        # Python File to clean the data                  
│   ├── exploratory_data_analysis/            # Exploratory Data Analysis Directory
│   │   ├── __init__.py  
│   │   └── exploratory_data_analyzer.py      # Python file to perform EDA                                              
│   ├── pipeline/                             # ML Pipeline Directory
│   │   ├── __init__.py   
│   │   └── model_pipeline.py                 # Python file to Create the pipeline                                               
│   └── utils/                                # Utility Functions Directory
│       ├── __init__.py                     
│       ├── audio_saver.py                    # Audio save and loading feature
│       ├── data_loader.py                    # Data save and loading feature
│       ├── logger.py                         # Logger Setup
│       ├── model_loader.py                   # Model save and loading feature
│       ├── pipeline_saver.py                 # Save the Pipeline
│       └── save_plot.py                      # Save the plot into particular directory       
└── web/
    ├── __init__.py  
    ├── pages/                                # Directory for the Different Streamlit Pages
    │   ├── __init__.py 
    │   ├── about.py                          # About Page
    │   ├── home.py                           # Home Page
    │   ├── monitor.py                        # Monitor Page
    │   └── speech_to_text.py                 # STT Page
    └── utils/                                # Directory for the utility functions of Streamlit Pages
        ├── __init__.py   
        └── database_manager.py               # Python file utility functions of the Web App
```
---

## 🔮 Future Work

To enhance the capabilities of **Affective AI** and scale it into a more robust emotion-aware system, we have outlined the development roadmap in the following phases:

### 🚀 **Phase 1: Immediate Enhancements** *(Timeframe: 1–2 months)*

* ✅ **Multi-language Support**: Integrate multilingual transcription using Whisper's multilingual capabilities.
* ✅ **User Authentication**: Add login/signup functionality for personalized tracking.
* ✅ **Emotion Confidence Scores**: Display confidence/probability of detected emotions.
* ✅ **Custom Dataset Upload**: Allow users to upload their own datasets for model training.
* ✅ **UI/UX Improvements**: Make UI more dynamic and responsive across devices.


### 🔧 **Phase 2: Intermediate Upgrades** *(Timeframe: 2–4 months)*

* ✅ **Contextual Emotion Tracking**: Track emotion changes across longer sessions or conversations.
* ✅ **Real-Time Audio Stream Analysis**: Detect emotion from live audio streams instead of recorded files.
* ✅ **Mobile Responsiveness**: Optimize the platform for mobile browsers and potential app integration.
* ✅ **Cloud Database Integration**: Move from local `data.db` to cloud-based storage (e.g., Firebase, MongoDB Atlas).
* ✅ **Model Fine-Tuning**: Train emotion classification models on larger and more diverse datasets for improved accuracy.


### 🌐 **Phase 3: Advanced Capabilities and Deployment** *(Timeframe: 4–6 months)*

* ✅ **Cross-Modal Emotion Detection**: Combine audio and facial expression analysis (audio + video input).
* ✅ **Dashboard Analytics**: Provide detailed user-based emotional analytics and trends over time.
* ✅ **Deployment as SaaS**: Launch Affective AI as a Software-as-a-Service (SaaS) platform for enterprises.
* ✅ **API Development**: Build RESTful APIs for third-party emotion detection integrations.
* ✅ **Voice Emotion Feedback Loop**: Allow users to manually verify/correct emotions to improve future model accuracy.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

<div align="center">

**Made by Priyam Pal - AI and Data Science Engineer**

[↑ Back to Top](#-affective-ai--understanding-emotions-through-text)

</div>
