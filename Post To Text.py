import requests
from bs4 import BeautifulSoup

#TODO - FIX AND ADD REDDIT API.

def scrape_reddit_post(post_url):
    response = requests.get(post_url)

    if response.status_code == 200:
        html_content = response.content
        soup = BeautifulSoup(html_content, "html.parser")
        
        post_title = soup.find("h1", class_="_eYtD2XCVieq6emjKBH3m")
        post_content = soup.find("div", class_="s1q9J60OPnBzNp1iHpyvd8")

        if post_title is not None:
            print(f"Title: {post_title.text}")
        else:
            print("Title not found")
        
        if post_content is not None:
            print(f"Content: {post_content.text}")
        else:
            print("Content not found")
    else:
        print("Failed to retrieve the page")

if __name__ == "__main__":
    post_url = "https://www.reddit.com/r/ironscape/comments/15mdegx/i_now_understand_why_people_call_cg_a_prison/?utm_source=share&utm_medium=web2x&context=3"
    scrape_reddit_post(post_url)
