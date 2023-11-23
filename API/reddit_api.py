from flask import Flask, jsonify, request
import requests

app = Flask(__name__)

@app.route('/get_reddit_data', methods=['GET'])
def get_reddit_data():
    topic = request.args.get('topic', default='', type=str)
    if not topic:
        return jsonify({"error": "Topic parameter is required"}), 400

    # Call Pushshift API to get Reddit data based on the chosen topic
    url = f'https://api.pushshift.io/reddit/search/submission/?q={topic}&size=10'
    response = requests.get(url)
    data = response.json()

    # Extract relevant information from the response
    relevant_data = [{'title': post['title'], 'url': post['url']} for post in data['data']]

    return jsonify(relevant_data)

if __name__ == '__main__':
    app.run(debug=True)
