import os
import logging
from flask import Flask, request, jsonify
from pyngrok import ngrok
from dotenv import load_dotenv
from sentiment import SentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
load_dotenv()

analyzer = SentimentAnalyzer()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    # if any(keyword in text.lower() for keyword in SPAM_KEYWORDS):
    #     return jsonify({"error": "Spam content detected"}), 400

    try:
        result = analyzer.predict(text)
        # return jsonify(result)
        return jsonify({
            "success": True,
            "result": result[0] if len(result) == 1 else result
        })
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("🚀 Starting server...")
    key = os.getenv("NGROK_KEY")
    if key:
        ngrok.set_auth_token(key)
        public_url = ngrok.connect(5000)
        print(f"🌍 Ngrok URL: {public_url}")
    app.run(host="0.0.0.0", port=5000, debug=False)