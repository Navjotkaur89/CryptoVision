import requests
import json
import streamlit as st 

st.cache_data
def fetch_data(limit=70):
    url = "https://api.livecoinwatch.com/coins/list"
    payload = json.dumps({
        "currency": "USD",
        "sort": "rank",
        "order": "ascending",
        "offset": 0,
        "limit": limit,
        "meta": True
    })
    headers = {
        'content-type': 'application/json',
        'x-api-key': '0eef1801-868a-4415-8b37-1f0a1bbb976a'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()

