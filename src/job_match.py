import streamlit as st
import webbrowser as wb
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
from weaviate.connect import ConnectionParams
from weaviate.util import generate_uuid5
import os
import pickle
import json

from weaviate_utils import load_data, create_collection
from llm_utils import extract_info

with open('../data/industries.pkl', 'rb') as f:
	industries_data = pickle.load(f)

with open('../data/roles.pkl', 'rb') as f:
	roles_data = pickle.load(f)

with open ('../data/abouts', 'rb') as file:
    example_summaries = pickle.load(file)

with open ('../data/states.pkl', 'rb') as file:
	states = pickle.load(file)

role_type_map = {'Flexible': 0, 'Full-Time': 1, 'Part-Time': 2, 'Internship': 3}
states = ['Flexible'] + states
location_map = {state: i for i, state in enumerate(states)}
remote_map = {'Flexible': 0, 'Remote': 1}

if 'summary_index' not in st.session_state:
	st.session_state.summary_index = -1
if 'role_type_selection' not in st.session_state:
	st.session_state.role_type_selection = 0
if 'location_selection' not in st.session_state:
	st.session_state.location_selection = 0
if 'remote_selection' not in st.session_state:
	st.session_state.remote_selection = 0

def retrieve_jobs(candidate_information:dict, reload_collection:bool=False, top_k=5):

	client = weaviate.connect_to_local(additional_config=wvc.init.AdditionalConfig(timeout=(60, 7500)))
	collection_name = 'Jobs'

	if reload_collection or not client.collections.exists(collection_name):
		jobs_collection = create_collection(client, collection_name=collection_name)
		load_data(jobs_collection, 'data/jobs.json', 100)
	else:
		jobs_collection = client.collections.get(collection_name)

	query = candidate_information['summary']

	vector = wvc.query.HybridVector.near_text(
		query=query, 
		move_to=wvc.query.Move(
			force=0.5, 
			concepts=roles)) if len(candidate_information['interested_roles']) > 0 else None

	filters = wvc.query.Filter.by_property("industry").contains_any(industries) if len(candidate_information['industries']) > 0 else None

	response = jobs_collection.query.hybrid(
		query=query,
		query_properties=['description^2', 'skills', 'industry'],
		fusion_type=wvc.query.HybridFusion.RELATIVE_SCORE,
		target_vector='description_vector',
		filters=filters,
		vector=vector,
		return_metadata=wvc.query.MetadataQuery(score=True),
		alpha=0.6,
		limit=50,
		auto_limit=4,
		rerank=wvc.query.Rerank(prop='description', query=query))

	client.close()

	return response.objects[:top_k]

def show_result(job):
	# output_box.markdown(f'---------------------------------------------------------------------')
	html_str = f"""
	<style>
	p.a {{
	  font: bold 20px "IBM Plex Sans";
	}}
	</style>
	<p class="a">{job.properties["title"]}</p>
	<p class="a">{job.properties["company_name"]}</p>
	"""
	output_box.markdown(html_str, unsafe_allow_html=True)
	output_box.markdown(f'---------------------------------------------------------------------')
	output_box.markdown(f'{job.properties["description"]}')
	output_box.markdown(f'Industry: {job.properties["industry"]}')
	output_box.markdown(f'Score: {job.metadata.score:.3f}')


def search_callback():

	if st.session_state.summary == '':
		return
	
	candidate_information = extract_info(st.session_state.summary)
	candidate_information['interested_roles'] = st.session_state.roles
	candidate_information['industries'] = st.session_state.industries
	candidate_information['role_type'] = st.session_state.role_type
	candidate_information['remote'] = st.session_state.remote
	candidate_information['location'] = st.session_state.location  
	jobs = retrieve_jobs(candidate_information, top_k=int(st.session_state.top_k))
	st.session_state.results_index = 0
	st.session_state.results = jobs
	c3.markdown(f'Role {st.session_state.results_index + 1}/{st.session_state.top_k}')

	job = jobs[st.session_state.results_index]
	show_result(job)


def autofilter_callback():

	if st.session_state.summary == '':
		return
	
	candidate_information = extract_info(st.session_state.summary)

	roles_inferred = [role for role in candidate_information['interested_roles'] if role in roles_data]
	industries_inferred = [industry for industry in candidate_information['industries'] if industry in industries_data]

	st.session_state.remote_selection = remote_map.get(candidate_information['remote'], 0)
	st.session_state.location_selection = location_map.get(candidate_information['location'], 0)
	st.session_state.role_type_selection = role_type_map.get(candidate_information['role_type'], 0)
	st.session_state.roles = roles_inferred
	st.session_state.industries = industries_inferred

	output_box.markdown('**Inferred**')
	if len(roles_inferred) > 0:
		output_box.markdown(f'Roles: {", ".join(roles_inferred)}\n')
	if len(industries_inferred) > 0:
		output_box.markdown(f'Industries: {", ".join(industries_inferred)}\n')
	output_box.markdown(f'Role Type: {st.session_state.role_type}\n')
	output_box.markdown(f'Remote: {st.session_state.remote}\n')
	output_box.markdown(f'Location: {st.session_state.location}\n')

def prev_example_callback():
	st.session_state.summary_index = max(0, st.session_state.summary_index - 1)
	st.session_state.summary = example_summaries[st.session_state.summary_index]
	cu1.markdown(f'Document: {st.session_state.summary_index + 1} / {len(example_summaries)}')


def next_example_callback():
	st.session_state.summary_index = min(len(example_summaries) - 1, st.session_state.summary_index + 1)
	st.session_state.summary = example_summaries[st.session_state.summary_index]
	cu1.markdown(f'Document: {st.session_state.summary_index + 1} / {len(example_summaries)}')

def prev_role_callback():
	if 'results' not in st.session_state:
		return
	k = max(0, st.session_state.results_index - 1)
	st.session_state.results_index = k
	show_result(st.session_state.results[k])
	c3.markdown(f'Role {st.session_state.results_index + 1}/{st.session_state.top_k}')


def next_role_callback():
	if 'results' not in st.session_state:
		return
	k = min(st.session_state.top_k - 1, st.session_state.results_index + 1)
	st.session_state.results_index = k
	show_result(st.session_state.results[k])
	c3.markdown(f'Role {st.session_state.results_index + 1}/{st.session_state.top_k}')

def apply_callback():
	if 'results' not in st.session_state:
		return

	wb.open_new_tab(st.session_state.results[st.session_state.results_index].properties['application_url'])

body = st.container()
body.title('Job Search')
cu1, _, cu2, cu3 = st.columns([8, 8, 6, 5])
body.write('Tell us about you, and let us find you some relevant roles!')
with cu2:
	st.button('Previous Example', type='secondary', on_click=prev_example_callback)
with cu3:
	st.button('Next Example', type='secondary', on_click=next_example_callback)

sidebar = st.sidebar
sidebar.title('Filters')
roles = sidebar.multiselect(label='Roles', key='roles', options=roles_data)
industries = sidebar.multiselect(label='Industries', key='industries', options=industries_data)
remote = sidebar.selectbox(label='In-Person / Remote', key='remote', index=st.session_state.remote_selection, options=remote_map.keys())
location = sidebar.selectbox(label='Location', key='location', index=max(0, st.session_state.location_selection - 1), options=location_map.keys())
role_type = sidebar.selectbox(label='Role Type', key='role_type', index=st.session_state.role_type_selection, options=role_type_map.keys())
top_k = sidebar.selectbox(label='Role Count', key='top_k', index=4, options=list(range(1, 21)))

summary = st.text_area('', height=500, key='summary')
c1, c2, c3, c5, c6 = st.columns([3,4, 6, 4, 4])

c1.button('Search', type='primary', on_click=search_callback)
c2.button('Auto Filter', type='secondary', on_click=autofilter_callback)
c5.button('Previous Role', type='secondary', on_click=prev_role_callback)
c6.button('Next Role', type='secondary', on_click=next_role_callback)

output_box = st.container(height=700)

_, _, _, cl2, _, _, _ = st.columns([1] * 7)
cl2.button('Apply', key='apply', on_click=apply_callback)


	