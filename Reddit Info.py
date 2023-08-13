import requests
from dotenv import load_dotenv
import praw
import os
import pandas as pd

#load .env variables
load_dotenv()


client_id = os.environ.get("REDDITCLIENTID")
secret_key = os.environ.get("REDDITSECRETKEY")
reddit_username = os.environ.get("REDDITUSERNAME")
reddit_password = os.environ.get("REDDITPASSWORD")
reddit_redirectURL= os.environ.get("REDDITREDIRECTURL")


reddit = praw.Reddit(client_id=client_id, client_secret=secret_key,redirect_url=reddit_redirectURL,user_agent=' RedTok app secretage001')


print(reddit.read_only)

# auth = requests.auth.HTTPBasicAuth(client_id,secret_key)

# data = {
#     'grant_type': 'password',
#     'username': reddit_username,
#     'password': reddit_password
# }

# #version of API
# headers = {'User-Agent': 'MyAPI/0.0.1'}

# #request Oauth token
# res = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data,headers=headers)

# print(res.json()['access_token'])