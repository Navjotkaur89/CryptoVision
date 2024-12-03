import requests

url = "https://cryptocurrency-news2.p.rapidapi.com/v1/cryptodaily"

def cryptonews():
    headers = {
	"x-rapidapi-key": "312617ff61msh2751704ca7e6b9cp13af57jsnbf25308174d7",
	"x-rapidapi-host": "cryptocurrency-news2.p.rapidapi.com"
}





    response = requests.get(url, headers=headers)
    #print(response.json())
    return response.json()
cryptonews()