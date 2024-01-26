
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import tldextract
import os

driver = webdriver.Chrome()

def get_screenshot(url,country):
    driver.get(url)
    driver.maximize_window()
    extracted = tldextract.extract(url)

    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
    driver.save_screenshot(f'./samples/{country}/{extracted.domain}.png')

def screen_loop(list,country):
    for url in list:
        get_screenshot(url,country)







