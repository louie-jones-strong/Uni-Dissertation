import os
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import sys


class Main():

	ConfigPath = 'config.json'
	ReportPath = 'report.json'
	PaperFolderPath = 'papers'

	def __init__(self, headless=False):

		# Load the config JSON file
		with open(self.ConfigPath) as f:
			self.Websites = json.load(f)

		# setup the webdriver
		options = webdriver.ChromeOptions()
		options.add_argument('disable-logging')
		options.add_argument('log-level=3')

		if headless:
			options.add_argument('headless')

		self.Browser = webdriver.Chrome(options=options)


		if os.path.exists(self.ReportPath):
			with open(self.ReportPath) as f:
				self.Report = json.load(f)
		else:
			self.Report = {}

		return

	def __del__(self):

		# close the browser
		self.Browser.quit()
		return


	def UpdateAllPapers(self):
		for paperName in self.Report:
			self.GetPaperReport(paperName)

		self.SaveReport()
		return

	def AddPaper(self, paperName):
		self.SaveReport()
		return





	def GetPaperReport(self, paperName):
		print(f"""Getting the report for {paperName}""")

		# check if the paper is already downloaded (completed in the report)
		if paperName in self.Report:
			return self.Report[paperName]

		# if not, download the paper
		paper_url = self.DownloadPaper(paperName)


		return

	def SaveReport(self):
		with open(self.ReportPath, 'w') as f:
			json.dump(self.Report, f, indent=4)

		return

	def DownloadPaper(self, paperName):
		report = {}

		for website in self.Websites:
			websiteInfo = self.Websites[website]

			print(f"""Searching on {website}""")

			# Extract the information from the websiteInfo dictionary
			url = websiteInfo.get('url')
			search_box_xpath = websiteInfo.get('search_box_xpath')
			search_button_xpath = websiteInfo.get('search_button_xpath')
			first_result_xpath = websiteInfo.get('first_result_xpath')

			self.Browser.get(url)
			time.sleep(3)
			search_box = self.Browser.find_element(By.XPATH,search_box_xpath)
			search_box.send_keys(paperName)
			search_button = self.Browser.find_element(By.XPATH, search_button_xpath)
			search_button.click()
			time.sleep(3)
			first_result = self.Browser.find_element(By.XPATH, first_result_xpath)
			paper_title = first_result.text
			first_result.click()
			time.sleep(3)
			paper_url = self.Browser.current_url


		return



if __name__ == '__main__':
	# get the command line arguments
	args = sys.argv


	headless = False
	paperName = "Mastering the game of Go with deep neural networks and tree search"


	main = Main(headless=headless)

	if paperName is not None: # just add one paper
		main.GetPaperReport(paperName)
	else: # updated all the papers in the report
		main.UpdateAllPapers()

	del main

	print('Done')

