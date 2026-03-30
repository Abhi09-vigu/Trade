import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = Flask(__name__)

SYSTEM_PROMPT = """You are an AI trading assistant.

Your task is to analyze the provided market data and generate a trade signal STRICTLY in the exact format below. DO NOT output any other text, explanations, or disclaimers.

🪙 [Symbol]
⏳ Expiration 5 minutes
✅ Entry at [Current Time]
[🟢 BUY / 🔴 SELL]

Use martingale if necessary 👇

1️⃣ MARTINGALE AT [Current Time + 5 minutes]
2️⃣ MARTINGALE AT [Current Time + 10 minutes]

Rules:
1. Determine BUY or SELL based on the input data (e.g. RSI, MA, Trend).
2. Calculate the Martingale times by adding 5 and 10 minutes exactly to the provided Current Time in HH:MM format.
3. Keep the exact emojis and phrasing shown above."""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return jsonify({"error": "Gemini API Key is missing in .env file."}), 500

    symbol = data.get('symbol')
    price = data.get('price')
    rsi = data.get('rsi')
    ma = data.get('ma')
    trend = data.get('trend')
    volume = data.get('volume')
    model = data.get('model', 'meta-llama/llama-3-8b-instruct')

    if not all([symbol, price, rsi, ma, trend, volume]):
        return jsonify({"error": "All market data fields are required."}), 400
    current_time_str = datetime.now().strftime("%H:%M")

    user_message = f"""========================
📊 INPUT DATA
=============
Symbol: {symbol}
Current Price: {price}
RSI: {rsi}
Moving Average: {ma}
Trend: {trend}
Volume: {volume}
Current Time: {current_time_str}"""
    try:
        genai.configure(api_key=api_key)
        modelInstance = genai.GenerativeModel('gemini-2.5-flash', system_instruction=SYSTEM_PROMPT)
        
        response = modelInstance.generate_content(
            user_message,
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
        
        return jsonify({"analysis": response.text})

    except Exception as e:
        return jsonify({"error": f"API request failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
