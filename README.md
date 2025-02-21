# SMS Spam Classification Project

This project is an SMS spam classification system that uses machine learning to classify SMS messages as either **Spam** or **Not Spam**. The system consists of a trained machine learning model and a Streamlit-based web interface.

---

## File Structure

```
sms-classification/
│
├── data/
│   ├── spam.csv          # Dataset containing SMS messages and labels
│
├── model/
│   ├── sms_model.pkl     # Trained Logistic Regression model
│   ├── vectorizer.pkl    # TF-IDF vectorizer
│
├── app/
│   ├── app.py            # Streamlit application for message classification
│
├── train_model.py        # Python script to train and save the model
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Prerequisites

- Python 3.8+
- `pip` for installing dependencies

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sms-classification.git
   cd sms-classification
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset (`spam.csv`) is placed in the `data/` directory.

---

## Usage

### **1. Train the Model**

Run the `train_model.py` script to train the Logistic Regression model:
```bash
python train_model.py
```

### **2. Start the Streamlit App**

Navigate to the `app/` directory and start the Streamlit server:
```bash
cd app
streamlit run app.py
```
The application will automatically open in your default web browser.

### **3. Use the Application**

1. Enter an SMS message in the text area
2. Click the **Classify** button
3. View the classification result with confidence scores
4. Expand "Message Details" to see probability distribution

---

## Features

- Real-time SMS classification
- Probability scores for predictions
- Interactive visualization of classification results
- User-friendly interface powered by Streamlit
- Detailed message analysis with probability distribution charts

## Dataset

The dataset (`spam.csv`) contains SMS messages labeled as `ham` (not spam) or `spam`. It has the following structure:

| Label | Message               |
|-------|-----------------------|
| ham   | Hello, how are you?   |
| spam  | WIN a $1000 gift card!|

---

## Key Files

### **1. `train_model.py`**

- Preprocesses the dataset
- Trains a Logistic Regression model using TF-IDF vectorization
- Saves the trained model and vectorizer

### **2. `app/app.py`**

- Streamlit application for SMS classification
- Loads the trained model and vectorizer
- Provides interactive interface with probability visualization

---

## Dependencies

Main requirements:
- Streamlit
- scikit-learn
- pandas
- numpy
- nltk
- matplotlib
- seaborn

Install using:
```bash
pip install -r requirements.txt
```

---

## Future Enhancements

- Add batch processing for multiple messages
- Implement model retraining functionality
- Add support for multiple languages
- Include more detailed analytics and visualizations
- Deploy the application on Streamlit Cloud
