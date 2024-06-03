# Author: Suprateem Banerjee [www.github.com/suprateembanerjee]

import ollama
import os
import requests
import json

# Used to verify whether an LLM Generated JSON meets required specifications
def verify_extraction(json_extraction):

	if set(json_extraction.keys()) != set(["skills", "location", "role_type", "interested_roles", "industries", "remote", "team_fit"]):
		return False

	if len(json_extraction['team_fit'].split(' ')) > 100:
		return False

	return True

# Extracts information from a given summary using an LLM, into a JSON
def extract_info(summary:str):

	prompt_template = f'''
	Given the following summary of a jobseeker: "{summary}" answer the following questions: 
	1. Enlist the main skills of the jobseeker.
	2. Is the jobseeker interested in working in a specific location? If yes, mention the full name of the US State for this location. If no, just say "Flexible".
	3. Is the jobseeker looking for a "Full-Time" role, an "Internship" role, or "Contract" role? If any of these, mention the type name. Else, say "Flexible".
	4. What are 3 role titles that the jobseeker might be interested in? Be curt, do not explain role.
	5. Does the jobseeker have experience or interest in specific industries? If so, list the industries, if not, say "Flexible". Do not explain, only list.
	6. Does the jobseeker have a strong preference for "Remote" work? If so, say "Remote", else say "Flexible". Do not explain, only list.
	7. Write a 100 word paragraph on the kind of team this person would excel at.
	Structure answers into a json that can be read using Python json.loads() using the keys "skills", "location", "role_type", "interested_roles", "industries", "remote" and "team_fit" respectively. 
	Do not include any other explanations or sentences in the output. Do not explain how to use it.
	'''

	json_output = None
	generation_count = 0

	while not json_output:

		generation_count += 1
		output = ollama.generate(model = "llama3", prompt = prompt_template)

		for x in output['response'].split('```'):
			try:
				json_output = json.loads(x)
				if verify_extraction(json_output):
					break
				else:
					json_output = None
			except:
				pass

	print(f'It took {generation_count} LLM call{"s" if generation_count > 1 else ""} to extract details.')
	json_output['summary'] = summary

	return json_output