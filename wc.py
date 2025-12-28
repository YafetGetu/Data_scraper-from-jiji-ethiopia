# from bs4 import BeautifulSoup

# with open('t.html', 'r') as html_file:
#     content = html_file.read()
    
#     soup = BeautifulSoup(content, 'lxml')
    
#     tag = soup.find_all('h3')
    
#     for info in tag:
#         print(info.text)


import requests

from bs4 import BeautifulSoup

#website url 
url = 'https://jiji.com.et/'

#sending http request to the website 

response = requests.get(url)

soup = BeautifulSoup(response.text, 'lxml')


soup = BeautifulSoup(response.text, 'lxml')

# Step 4: Find all titles and prices
titles = soup.find_all('div', class_='b-advert-title-inner qa-advert-title b-advert-title-inner--div')
prices = soup.find_all('div', class_='qa-advert-price')

# Step 5: Print them together
for title, price in zip(titles, prices):
    print(f"{title.text.strip()} - {price.text.strip()}")