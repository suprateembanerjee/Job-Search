# Author: Suprateem Banerjee [www.github.com/suprateembanerjee]

import ollama
import os
import requests
import json

# Used to verify whether an LLM Generated JSON meets required specifications
# def verify_extraction(json_extraction):

# 	if set(json_extraction.keys()) != set(["skills", "location", "role_type", "interested_roles", "industries", "remote", "team_fit"]):
# 		return False

# 	if len(json_extraction['team_fit'].split(' ')) > 100:
# 		return False

# 	return True

# Used to verify whether an LLM Generated JSON meets required specifications
def verify_extraction(json_extraction, checks):

	# Parameters not validated
	checks['location'] = False
	checks['skills'] = False

	if set(json_extraction.keys()) != set(checks):
		return {check:True for check in checks}

	if checks['team_fit'] and len(json_extraction['team_fit'].split(' ')) < 100:
		checks['team_fit'] = False

	if checks['remote'] and json_extraction['remote'] in ['Remote', 'Flexible']:
		checks['remote'] = False
	
	if checks['role_type'] and json_extraction['role_type'] in ['Full-Time', 'Internship', 'Contract', 'Flexible']:
		checks['role_type'] = False
	
	if checks['interested_roles']:
		flag = False
		for role in json_extraction['interested_roles']:
			if role[-3:] not in ['yst', 'ist', 'ant'] and role[-2:] not in ['er']:
				flag = True
		checks['interested_roles'] = flag
	
	if checks['industries']:
		flag = False
		for industry in json_extraction['industries']:
			prompt = f'''
			We are trying to identify whether a term is composed of full english words, or abbreviations.

			Here are a few examples:
			"Healthcare" -> "Full",
			"Home Science" -> "Full",
			"Defense" -> "Full",
			"Intelligence" -> "Full",
			"Education Technology" -> "Full",
			"EdTech" -> "Abbreviated",
			"Fintech" -> "Abbreviated",
			"CompSci" -> "Abbreviated"

			This is a term: {industry}. Is this an abbreviated term or full form?
			Answer in "Abbreviated" or "Full" accordingly. Do not explain.
			'''
			response = ollama.generate(model = "llama3", prompt=prompt)['response']
			if 'Full' not in response:
				flag = True
				break
		checks['industries'] = flag

	return checks

# # Extracts information from a given summary using an LLM, into a JSON
# def extract_info(summary:str):

# 	prompt_template = f'''
# 	Given the following summary of a jobseeker: "{summary}" answer the following questions: 
# 	1. Enlist the main skills of the jobseeker.
# 	2. Is the jobseeker interested in working in a specific location? If yes, mention the full name of the US State for this location. If no, just say "Flexible".
# 	3. Is the jobseeker looking for a "Full-Time" role, an "Internship" role, or "Contract" role? If any of these, mention the type name. Else, say "Flexible".
# 	4. What are 3 role titles that the jobseeker might be interested in? Be curt, do not explain role.
# 	5. Does the jobseeker have experience or interest in specific industries? If so, list the industries, if not, say "Flexible". Do not explain, only list.
# 	6. Does the jobseeker have a strong preference for "Remote" work? If so, say "Remote", else say "Flexible". Do not explain, only list.
# 	7. Write a 100 word paragraph on the kind of team this person would excel at.
# 	Structure answers into a json that can be read using Python json.loads() using the keys "skills", "location", "role_type", "interested_roles", "industries", "remote" and "team_fit" respectively. 
# 	Do not include any other explanations or sentences in the output. Do not explain how to use it.
# 	'''

# 	json_output = None
# 	generation_count = 0

# 	while not json_output:

# 		generation_count += 1
# 		output = ollama.generate(model = "llama3", prompt = prompt_template)

# 		for x in output['response'].split('```'):
# 			try:
# 				json_output = json.loads(x)
# 				if verify_extraction(json_output):
# 					break
# 				else:
# 					json_output = None
# 			except:
# 				pass

# 	print(f'It took {generation_count} LLM call{"s" if generation_count > 1 else ""} to extract details.')
# 	json_output['summary'] = summary

# 	return json_output

def extract_info(summary:str):

	questions = dict(skills='Enlist the main skills of the jobseeker.',
                 location='Is the jobseeker interested in working in a specific location? If yes, mention the full name of the US State for this location. If no, just say "Flexible".',
                 role_type='Is the jobseeker looking for a "Full-Time" role, an "Internship" role, or "Contract" role? If any of these, mention the type name. Else, say "Flexible".',
                 interested_roles='What are 3 role titles that the jobseeker might be interested in? Make sure these are tangible roles, not domains. Be curt, do not explain role.',
                 industries='Does the jobseeker have experience or interest in specific industries? If so, list the industries using full form of the industry names, without using abbreviations. If it is an abbreviation, convert it to full form (for example "EdTech" to "Education Technology"). If no specific industries are found, say "Flexible". Do not explain, only list.',
                 remote='Does the jobseeker have a strong preference for "Remote" work? If so, say "Remote", else say "Flexible". Do not explain, only list.',
                 team_fit='Write a 100 word paragraph on the kind of team this person would excel at.')

	remaining_questions = questions
	checks = {question: True for question in questions}

	json_object = {}

	while any(checks.values()):

		remaining_questions = {key: value for key, value in questions.items() if checks[key]}
		formatted_questions = '\n'.join([f'{i}: {question}' for i, question in enumerate(remaining_questions.values(), start=1)])

		prompt = f'''
		Given the following summary of a jobseeker: "{summary}" answer the following question(s):
		{formatted_questions}
		Structure answers into a json that can be read using Python json.loads() using the following key(s): {', '.join([f"{key}" for key in remaining_questions.keys()])}. 
		Do not include any other explanations or sentences in the output. Do not explain how to use it.
		''' 

		output = ollama.generate(model = "llama3", prompt = prompt)

		for x in output['response'].split('```'):
			try:
				json_output = json.loads(x)
				for key in json_object:
					json_output[key] = json_object[key]
				checks = verify_extraction(json_output, checks)
				for key, check in checks.items():
					if not check:
						json_object[key] = json_output[key]
				if not any(checks.values()):
					break
			except:
				pass
	
	json_object['summary'] = summary
	return json_object