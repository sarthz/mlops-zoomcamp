# import predict
import requests

ride = {"PULocationID": 10 , 
        "DOLocationID": 50 }

url = 'http://127.0.0.1:9696/predict'
response = requests.post(url, json=ride)
# response.raise_for_status()  # raises exception when not a 2xx response
# if response.status_code != 204:
#     print(response.json())
print(response.json())

# pred = predict.predict(ride)
# print(pred)