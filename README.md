# Test Screenshot to Text Converter

This tool extracts text from multiple-choice test screenshots and identifies the correct answer (highlighted in green).

## Setup

1. **Install Python:** Ensure you have Python installed (3.8+ recommended).

2. **Install Dependencies:**
   Run the following command in your terminal to install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

To start the application, run:

```bash
streamlit run app.py
```

This will launch the app in your default web browser (usually at http://localhost:8501).

## Usage

1. Open the app in your browser.
2. Upload a screenshot of a test question (supports JPG, PNG, WEBP).
3. Wait for the processing to finish.
4. View the extracted text and the identified correct answer.
5. You can edit the text in the provided text area before copying it.
