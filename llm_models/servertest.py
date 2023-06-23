from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def process_request():
    data = request.json
    # Process the data received in the request
    # Perform any necessary computations or tasks
    response = {'message': data['word']}
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=8000)