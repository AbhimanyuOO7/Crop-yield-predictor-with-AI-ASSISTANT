from flask import Flask, request, render_template
import numpy as np
import pickle
import google.generativeai as genai
import pandas as pd

# Load dataset once
df = pd.read_csv("yield_df.csv")
area_options = sorted(df["Area"].unique())
item_options = sorted(df["Item"].unique())

# Load models
dtr = pickle.load(open('C:/Users/ABHIMANYU.M.B/Desktop/Crop Predicter AI/Crop_Yield_Prediction-main/dtr.pkl', 'rb'))
preprocessor = pickle.load(open('C:/Users/ABHIMANYU.M.B/Desktop/Crop Predicter AI/Crop_Yield_Prediction-main/preprocesser.pkl', 'rb'))

# Configure Gemini API
genai.configure(api_key="AIzaSyBrYfsMZL4ZKwvqf2T3SO6eK_QEJpzMkUQ")  # Replace with your Gemini API key
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", area_options=area_options, item_options=item_options)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        return render_template("index.html", prediction=prediction[0][0], area_options=area_options, item_options=item_options)

    except Exception as e:
        error_message = f"""
        🚫 <strong>Prediction Failed:</strong> {str(e)}<br><br>
        ⚠️ Please check your input values for correctness.<br>
        💬 Try using the <strong>AI Assistant</strong> for help with valid Area and Item names.
        """
        return render_template("index.html", prediction=None, error=error_message, area_options=area_options, item_options=item_options)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == 'POST':
        user_input = request.form['user_input']
        prompt = f"""
        I am a farmer from {user_input}. Please suggest the best crops to grow and give 2 short tips to improve crop yield in this area during the current season.
        """

        try:
            response = gemini_model.generate_content(prompt)
            advice_text = response.text
        except Exception as e:
            advice_text = f"Error getting advice: {e}"

        return render_template("chatbot.html", advice=advice_text)

    # GET request (just show the page)
    return render_template('chatbot.html')

if __name__ == "__main__":
    app.run(debug=True)
