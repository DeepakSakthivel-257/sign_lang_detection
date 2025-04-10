# Sign Language Recognition System

A real-time sign language interpreter using OpenCV, MediaPipe, and scikit-learn that supports English alphabets.

## Features

- 📷 Real-time hand sign detection
- 🌍 Multi-language support (English/Tamil/Hindi)
- 🤖 Machine learning model training pipeline
- 📊 Data collection and preprocessing tools

### Prerequisites
- Python 3.7-3.10 (recommended)
- Webcam

### Setup
```bash
# Clone repository
git clone https://github.com/DeepakSakthivel-257/sign_lang_detection.git
cd sign_lang_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r req lib.txt


PROJECT STRUCTURE

sign-language/
├── data/                  # Training datasets
│   ├── raw/              # Raw collected images
│   └── processed/        # Processed numpy arrays
├── models/               # Saved ML models
├── notebooks/            # Jupyter notebooks for experimentation
├── src/
│   ├── data_collector.py # Data collection script
│   ├── train_model.py    # Model training script
│   └── interpreter.py    # Real-time prediction script
└── requirements.txt
