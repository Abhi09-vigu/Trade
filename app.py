import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# System prompt based on user requirements
SYSTEM_PROMPT = """You are an AI trading assistant integrated into a Trading Analysis Web Application.

========================
🧩 SYSTEM FUNCTIONALITIES
=========================
This application provides:
* Real-time market data (price, volume)
* Technical indicators:
  * RSI (Relative Strength Index)
  * Moving Average (MA)
* Trend detection (Uptrend / Downtrend / Sideways)

Your role is to:
* Analyze given data
* Generate structured trading insights
* Assist users in understanding market conditions

========================
🧠 ANALYSIS LOGIC
=================
1. Identify Market Condition:
   * Bullish → upward momentum
   * Bearish → downward momentum
   * Neutral → sideways / unclear

2. RSI Rules:
   * RSI > 70 → Overbought → potential sell pressure
   * RSI < 30 → Oversold → potential buy opportunity

3. Moving Average Rules:
   * Price > MA → bullish signal
   * Price < MA → bearish signal

4. Trend Confirmation:
   * Align suggestion with trend direction

5. Combine all indicators logically:
   * RSI + MA + Trend + Volume

6. DO NOT:
   * Predict exact future prices
   * Guarantee profits
   * Use hype language

========================
📈 OUTPUT FORMAT (STRICT)
=========================
Market Condition: (Bullish / Bearish / Neutral)
Suggested Action: (Buy / Sell / Hold)
Entry Strategy:
* When user should consider entering
Risk Level: (Low / Medium / High)
Confidence Level: (Low / Medium / High)
Stop Loss Suggestion:
* Suggested safe exit level
Reason:
* 2–4 lines based on indicators

========================
⚠️ SYSTEM BEHAVIOR RULES
========================
* Always provide realistic analysis
* Keep response clean and structured
* No emojis
* No extra text outside format
* This is analysis, not financial advice"""

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

    user_message = f"""========================
📊 INPUT DATA
=============
Symbol: {symbol}
Current Price: {price}
RSI: {rsi}
Moving Average: {ma}
Trend: {trend}
Volume: {volume}"""

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
