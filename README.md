<div align="center">

# 🤖 Affective AI — Understanding Emotions Through Text

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenAI](https://img.shields.io/badge/OpenAI-Whisper-blue.svg)](https://openai.com/research/whisper)
[![XGBoost](https://img.shields.io/badge/XGBoost-Integrated-orange.svg)](https://xgboost.ai/)

*Empowering machines to understand human emotions through textual data.*

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

```bash
# Clone the repository
git clone https://github.com/priyam-hub/Affective-AI.git

# Navigate into the directory
cd Affective-AI

# Install required dependencies
pip install -r requirements.txt
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

### 🔌 Core Technologies

- **Python 3.9+**: Primary programming language.
- **Streamlit**: Framework for building interactive web applications.
- **OpenAI Whisper**: Advanced speech recognition model.
- **XGBoost**: Optimized gradient boosting library for classification tasks.
- **Pandas & NumPy**: Data manipulation and numerical operations.
- **Plotly & Altair**: Interactive data visualization libraries.

---

## 📁 Project Structure

```plaintext
Affective-AI/
├── app.py                 # Main application script
├── requirements.txt       # List of dependencies
├── config/
│   └── config.py          # Configuration settings
├── src/
│   ├── utils/
│   │   ├── logger.py      # Logging utility
│   │   └── model_loader.py# Model loading functions
│   ├── audio_recorder/
│   │   └── record_audio.py# Audio recording functionalities
│   └── audio_transcriber/
│       └── transcribe_audio.py # Audio transcription logic
├── web/
│   └── pages/
│       ├── home.py        # Home page layout
│       ├── monitor.py     # Monitoring dashboard
│       └── about.py       # About section
└── README.md              # Project documentation
```

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

<div align="center">

**Affective AI — Bridging the Gap Between Text and Emotion**

[↑ Back to Top](#-affective-ai--understanding-emotions-through-text)

</div>