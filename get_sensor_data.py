import requests
import time
import os


def get_gas_data():
    url = 'http://192.168.12.149/get_gas'
    response = requests.get(url)
    res_json = response.json()
    return res_json

def get_temperature_humidity():
    url = 'http://192.168.12.149/get_temp_hum'
    response = requests.get(url)
    res_json = response.json()
    return res_json


if __name__ == '__main__':
    while True:
        gas_data = get_gas_data()
        temp_hum_data = get_temperature_humidity()
        print(gas_data)
        print(temp_hum_data)
        time.sleep(1)
        os.system('clear')