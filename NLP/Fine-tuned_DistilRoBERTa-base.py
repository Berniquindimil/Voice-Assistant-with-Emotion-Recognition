from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Use Hugging Face's pipeline for emotion classification
classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

# Predefined responses based on detected emotions
responses = {
    'joy': "I'm glad you're feeling happy! 😊 How can I assist you?",
    'sadness': "I'm really sorry you're feeling down. 😔 Want to talk about it?",
    'anger': "It sounds like you're feeling upset. 😠 How can I help you calm down?",
    'fear': "I'm here for you. It's okay to be scared sometimes. 😟 What can I do to help?",
    'disgust': "I sense you're feeling disgusted. 😒 Is everything okay?",
    'surprise': "Wow! It seems like something surprised you. 😲 Tell me more!",
    'neutral': "I see you're neutral. Let me know if there's anything you'd like to talk about!",
    'Uncertain': "I'm not sure how to react, could you rephrase it?"
}

# Prediction function: takes text input and returns the predicted emotion
def predict_emotion(text):
    # Use the classifier to predict the emotion
    result = classifier(text)
    predicted_emotion = result[0]['label']  # Get the predicted emotion label
    confidence = result[0]['score']  # Get the confidence score

    # Set a threshold for the confidence (e.g., 70%)
    if confidence > 0.7:
        return predicted_emotion
    else:
        return "Uncertain"


# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    user_message = None
    bot_reply = None
    if request.method == 'POST':
        # Get the text from the form
        user_message = request.form['text']
        
        # Make the emotion prediction
        predicted_emotion = predict_emotion(user_message)
        
        # Get the appropriate response from the bot
        bot_reply = responses.get(predicted_emotion, responses['Uncertain'])

    # Render the template and pass the conversation (user_message and bot_reply)
    return render_template('index.html', user_message=user_message, bot_reply=bot_reply)

# Run the app if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True, port=5001)

