import requests # for http requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import re
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.common.action_chains import ActionChains

def URL_builder(HEAD, TICKER, TAIL = None):
    if TAIL == None: # SeekingAlpha's url structure
        URL = HEAD + TICKER
    else: # MarketWatch's url structure
        URL = HEAD + TICKER + TAIL
    return URL



def Timestamps_Cleaning(timestamps):

    pattern = r'(\w+\.?) (\d{1,2}), (\d{4}) at (\d{1,2}):(\d{2}) ([APap]\.m\.) ET'
    w = 0
    timestamps_list = []
    # Scrape and clean up the timestamps
    for y in timestamps:
        match = re.match(pattern, y.lstrip().rstrip())

        if match:
            # Extract components from the regex match
            month_str, day_str, year_str, hour_str, minute_str, am_pm = match.groups()

            # Convert month string to a numeric month (e.g., 'Sep.' to '09')
            month = datetime.strptime(month_str, "%b.").strftime("%m")

            # Padding with leading zero if not present in hour_str
            if len(hour_str) != 2:
                hour_str = '0' + hour_str
            else:
                hour_str

            # a.m. and p.m. conversion to 24 hours due to regex problems with strings
            if am_pm.lower() == 'p.m.' and not (hour_str == '12'):
                hour_str = str(int(hour_str) + 12)
            elif am_pm.lower() == 'a.m.' and hour_str == '12':
                hour_str = str(00)
            else:
                hour_str

            # Combine components into a new datetime string
            formatted_datetime_str = f"{year_str}-{month}-{day_str} {hour_str}:{minute_str}"

            # Parse the formatted datetime string into a datetime object
            timestamps_list.append(datetime.strptime(formatted_datetime_str, '%Y-%m-%d %H:%M'))

            w += 1

            print(w)
        else:
            print("Datetime string does not match the expected format.")

    return timestamps_list

def NewScraper(TICKER,
                  HEAD,
                  father_tag,
                  father_attr,
                  father_attr_name,
                  headline_tag,
                  headline_attr,
                  headline_attr_name,
                  time_tag,
                  time_attr,
                  time_attr_name,
                  author_tag,
                  author_attr,
                  author_attr_name,
                  scroll_step,
                  n_headlines,
                  TAIL = None):

    URL = URL_builder(HEAD, TICKER, TAIL)

    # Set up a headless browser (you may need to install a WebDriver for your browser)
    driver = webdriver.Chrome()

    # Load the webpage
    driver.get(URL)

    # Create an empty list to store the headlines
    headlines_list = []

    # Create an empty list to store the timestamps
    timestamps_list = []

    # Create an empty list to store the authors
    authors_list = []

    # Create a set to keep track of seen tuples of headlines and timestamps
    seen_tuples = set()

    vertical_ordinate = 100


    while True:

        try:
            # Re-locate the scrollable element
            element = driver.find_element(By.CSS_SELECTOR, f"{father_tag}[{father_attr}='{father_attr_name}']")

            # Scroll down (to collect the headlines a manual scroll down in the MarketWatch headlines section is needed)
            driver.execute_script("arguments[0].scrollTop = arguments[1]", element, vertical_ordinate)
            vertical_ordinate += scroll_step
            time.sleep(1)


            # Parse the newly loaded HTML content
            HTMLcode = BeautifulSoup(element.get_attribute('innerHTML'), "html.parser")

            # Extract headlines
            headlines = [y.text.lstrip().rstrip() for y in
                         HTMLcode.find_all(headline_tag, {headline_attr: headline_attr_name})]


            # Extract timestamps
            timestamps = [y.text.lstrip().rstrip() for y in
                         HTMLcode.find_all(time_tag, {time_attr: time_attr_name})]

            # Extract authors
            authors = [y.text.lstrip().rstrip() for y in
                          HTMLcode.find_all(author_tag, {author_attr: author_attr_name})]

            scraped_tuples = []

            for p, j, g in zip(headlines, timestamps, authors):
                scraped_tuples.append((p, j, g))


            # Check for new tuples of headlines and timestamps
            new_tuples = [h for h in scraped_tuples if h not in seen_tuples]

            # Add the new tuples to the set of seen tuples
            seen_tuples.update(new_tuples)

            for tup in new_tuples:
                print(tup)

                # Append the new headlines to the list
                headlines_list.append(tup[0])

                # Append the new timestamps to the list
                timestamps_list.append(tup[1])

                # Append the new authors to the list
                authors_list.append(tup[2])

            if len(headlines_list) > n_headlines:
                break

        except Exception as e:
            # Handle any exceptions, but it's important to investigate why they occur
            print(f"An exception occurred: {str(e)}")


    driver.quit()

    return pd.DataFrame({'sentence': headlines_list, 'timestamp': timestamps_list, 'author': authors_list})
    # return headlines_list, timestamps_list

