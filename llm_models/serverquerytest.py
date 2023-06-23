import requests
import json

def send_post_request(query):
    url = "http://localhost:5000/generate-text"  # Replace with your API endpoint URL

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "text": query
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        result = response.json()
        # Process the result as needed
        return result
    else:
        print("Request failed with status code:", response.status_code)
        return None

# Example usage
query = "somebody who controls an organisation or group of people"
response = send_post_request(query)
if response is not None:
    print("Response:", response)
