# SMS Spam Classification Project

This project is an SMS spam classification system that uses machine learning to classify SMS messages as either **Spam** or **Not Spam**. The system consists of a trained machine learning model, a Flask-based backend, and a user-friendly frontend interface.

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
│   ├── static/
│   │   ├── styles.css    # CSS for styling the website
│   ├── templates/
│   │   ├── index.html    # HTML frontend for user interaction
│   ├── app.py            # Flask application for message classification
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

Run the `train_model.py` script to train the Logistic Regression model and save the vectorizer:
```bash
python train_model.py
```
This will generate the following files in the `model/` directory:
- `sms_model.pkl` (trained model)
- `vectorizer.pkl` (TF-IDF vectorizer)

### **2. Start the Flask App**

Navigate to the `app/` directory and start the Flask server:
```bash
cd app
python app.py
```
The application will be available at `http://127.0.0.1:5000/`.

### **3. Use the Website**

1. Open the application in a web browser.
2. Enter an SMS message in the input box.
3. Click the **Classify** button.
4. View the classification result (Spam or Not Spam).

---

## Dataset

The dataset (`spam.csv`) contains SMS messages labeled as `ham` (not spam) or `spam`. It has the following structure:

| Label | Message               |
|-------|-----------------------|
| ham   | Hello, how are you?   |
| spam  | WIN a $1000 gift card!|

---

## Key Files

### **1. `train_model.py`**

- Preprocesses the dataset.
- Trains a Logistic Regression model using TF-IDF vectorization.
- Saves the trained model and vectorizer.

### **2. `app/app.py`**

- Flask application to classify SMS messages.
- Loads the trained model and vectorizer.
- Provides an API endpoint (`/classify`) for message classification.

### **3. `app/templates/index.html`**

- Frontend interface for entering SMS messages and displaying classification results.

### **4. `app/static/styles.css`**

- CSS styles for the website.

---

## Dependencies

The required Python libraries are listed in `requirements.txt`:

```
Flask==2.3.2
scikit-learn==1.3.0
```

Install them using:
```bash
pip install -r requirements.txt
```

---

## Future Enhancements

- Add support for multiple languages.
- Improve model accuracy with advanced techniques like deep learning.
- Deploy the application on cloud platforms like AWS or Heroku.
