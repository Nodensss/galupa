# Test Screenshot to Text Converter

This tool extracts text from multiple-choice test screenshots and identifies the correct answer (highlighted in green).

## 🚀 Getting Started (from GitHub)

Follow these steps to download and run the project on your computer.

### 1. Prerequisites
- **Python:** Make sure you have Python installed (version 3.8 or higher). You can download it from [python.org](https://www.python.org/downloads/).
- **Git:** Make sure Git is installed to clone the repository. [Download Git](https://git-scm.com/downloads).

### 2. Clone the Repository
Open your terminal (Command Prompt, PowerShell, or Terminal) and run:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```
*(Replace `YOUR_USERNAME/YOUR_REPO_NAME` with the actual URL of this repository)*

### 3. Set up a Virtual Environment (Recommended)
It's best practice to use a virtual environment to avoid conflicts with other Python projects.

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install the required libraries:

```bash
pip install -r requirements.txt
```

### 5. Run the Application
Start the Streamlit app:

```bash
streamlit run app.py
```

The application should automatically open in your default web browser at `http://localhost:8501`.

## Usage

1. Open the app in your browser.
2. Upload a screenshot of a test question (supports JPG, PNG, WEBP).
3. Wait for the processing to finish.
4. View the extracted text and the identified correct answer.
5. You can edit the text in the provided text area before copying it.
