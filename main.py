from flask import Flask, request,jsonify
from flask_cors import CORS
from llm import *
import os
app = Flask(__name__)
CORS(app)


@app.route('/llm/chat',methods=['POST'])
async def chat_endpoint():
    """
    FastAPI endpoint to process a single user input and return chatbot response.
    """
    try:
       
        chatbot = ChatBot1(
            
            weather_api_key=os.getenv("WEATHER_API"),google_places_api_key=os.getenv("GOOGLE_PLACES_API_KEY"),mongodb_link=os.getenv("MONGODB"), publisher_id=os.getenv('PARTNERIZE_PUBLISHER_ID'),APPLICATION_KEY=os.getenv('PARTNERIZE_APPLICATION_KEY'),USER_API_KEY=os.getenv('PARTNERIZE_USER_API_KEY'),
            # hyper_browser_api_key=os.getenv('HYPER_BROWSER_API_KEY')
           
        )
        # chatbot = ChatBot(
            
        #     weather_api_key=os.getenv("WEATHER_API"),google_places_api_key=os.getenv("GOOGLE_PLACES_API_KEY"),mongodb_link=os.getenv("MONGODB"), publisher_id=os.getenv('PARTNERIZE_PUBLISHER_ID'),APPLICATION_KEY=os.getenv('PARTNERIZE_APPLICATION_KEY'),USER_API_KEY=os.getenv('PARTNERIZE_USER_API_KEY'),
        #     hyper_browser_api_key=os.getenv('HYPER_BROWSER_API_KEY')
           
        # )
        data  = request.get_json()
        user_input = data.get("user_input", "")
       
        # Get response
        response = await chatbot.chat2(user_input)
        print(f"\n \n\nResponse: {response}\n \n \n ")
        return jsonify({
            "response": response,
            
        })

    except Exception as e:

        return jsonify({
            "error": str(e),
            "history": []
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)


