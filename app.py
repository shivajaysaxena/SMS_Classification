import streamlit as st
import pickle
import os

# Load the pre-trained model and vectorizer
with open('model/sms_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('model/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# ...existing code...
def classify_message(message):
    # Vectorize the input message
    vectorized_message = vectorizer.transform([message])
    # Predict and return result
    prediction = model.predict(vectorized_message)[0]
    probability = model.predict_proba(vectorized_message)[0]
    return prediction, probability

# Set up the Streamlit page
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📱")

# Add a title and description
st.title("SMS Spam Classification")
st.markdown("""
This application helps you identify whether a text message is spam or not.
Enter your message below and click the 'Classify' button!
""")

# Create the input text area
message = st.text_area("Enter the message to classify:", height=100)

if st.button("Classify"):
    if message.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Get prediction and probability
        prediction, probability = classify_message(message)
        
        # Display result with probability
        if prediction == 1:
            st.error(f"📵 This message is likely SPAM (Confidence: {probability[1]:.2%})")
        else:
            st.success(f"✅ This message is likely NOT SPAM (Confidence: {probability[0]:.2%})")
        
        # Display message details
        with st.expander("Message Details"):
            st.write("**Input Message:**")
            st.write(message)
            st.write("**Probability Distribution:**")
            st.bar_chart({
                "Not Spam": probability[0],
                "Spam": probability[1]
            })

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit • SMS Spam Classification Project")
