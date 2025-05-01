import requests


response = requests.post(API_URL, headers=headers, json=payload, proxies=proxies, verify=True)
