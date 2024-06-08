# Gig Match

A Hybrid Retrieval System that matches job postings to candidates, based on their professional summaries. A demo video is available [here](https://youtu.be/f2V7B0KeHGs).

## Dataset
LinkedIn Job Postings Dataset [https://www.kaggle.com/datasets/arshkon/linkedin-job-postings] using the following code to scrape live LinkedIn data: https://github.com/ArshKA/LinkedIn-Job-Scraper

The dataset contains various tables regarding companies and jobs, but the meat of it lies in postings.csv. It contains 100k+ roles, but only ~87k of them are based in the US and have an application link, which I have used for my analysis.
After performing some Exploratory Data Analysis (refer `eda.ipynb`), I narrowed down the dataset to the following fields of interest for this project:
- job_id
- company_name
- title
- description
- location
- type
- remote
- skills
- industry
- application_url

We are going to use these fields in the following manner:

Hybrid Semantic Similarity Search: `title`, `description`, `skills`, `industry`\
Filtering: `location`, `type`, `remote`, `industry`\
UI: `company_name`, `application_url`

This data is stored in `jobs.json`.

From `industry` and `title`, I also generatively extracted the most commonly occurring roles and industries, which I used in my Streamlit UI. More on this later.

I also collected a sample of 27 professional summaries from some of my connections on LinkedIn. This list of strings is pickled into a file called `abouts`.\
I used `Llama3.8b` to extract some details from these summaries, namely the following:
- skills `list[str]`: A list of (technical) skills mentioned in the summary.
- location `str`: If there is a strong location preference, otherwise "Flexible".
- role_type `str`: If there is a strong preference for "Full-Time" / "Part-Time" / "Internship", else "Flexible".
- interested_roles `list[str]`: List of roles this candidate is interested in.
- industries `list[str]`: List of industries if the candidate has a specialization.
- remote `bool`: If the candidate is strongly in favor of "Remote", else "Flexible".
- team_fit `str`: A short 100-word summary of what kind of team this candidate would be a good fit for.
- summary `str`: The original summary.

This data is then stored in `extracted_summaries.json`.

## Semantic Similarity

Three separate vector indices exist to map job summaries, role titles, and industry titles. The first is used to map roles with candidates' professional summaries, whereas the role and industry indices enable automatic filtering of roles.
The similarity score between a job and a candidate's summary is evaluated through **Hybrid Semantic Similarity** before being passed into a **Re-Ranker**.

### Keyword Search

In Weaviate the default keyword search algorithm is BM25F, which is the Best Match 25 algorithm along with weighted field scores. In this project, a higher weightage is placed on `description` while also considering `skills` and `industry` for keyword search.

### Vector Search

The `nomic-embed-text` model from Nomic AI is used as the vectorizer for the only named vector in the jobs index. Not only are the vectors much smaller in dimension than llama3 (which was also experimented with), but they also supposedly have the functionality to specify the embedding dimensions between 64 and 768. 
However, since the vectors are *Binary Quantization*, the maximum embedding dimension is selected, which is 768.

HNSW Indexing is preferred due to its higher recall performance when dealing with a large volume of vectors (in our case, 87000). This seemed reasonable because this system is static, and new jobs won't be inserted daily. For a system such as this to be real-time, HNSW may be a sub-optimal choice as insertions in 
Hierarchical Indexes are costly (the whole Index needs to be remade). In this case, I would prefer Flat Indexing with Binary Quantization instead, as BQ (especially when coupled with SIMD-compatible distance metrics) might offer fast enough performance to not require indexing.

In this case, a SIMD-optimized distance metric, the L2 Distance, is used. Dynamic Indexing would have been an alternative, in which case the Index would switch from Flat to HNSW, but since we already know the total count to be high (default switch is around 10000, while we have 87000 documents), HNSW made more sense.

### Fusion
Having experimented with `RANKED` and `RELATIVE` fusion methods, I chose to use `RELATIVE` for my project as it offered better results. I parameterized the fusion using `alpha=0.6` which offered the best performance for my Hybrid Search.

### Re-Ranker

Before re-ranking, a vector alignment on the semantic search results is performed using the candidate's `interested_roles`. At this stage, the retrieval would have fetched a maximum of 50 vectors, but usually less since only the first 4 groups are fetched (according to the score) using `auto_limit=4`.

Following that, a cross-encoder model by *Cohere* named `rerank-english-v3.0` is used to re-rank the <50 results based on the role's `description`. This early-interaction mechanism vastly improved the search results for the top 5 (or any number, between 1 and 20) highest-ranked roles for a candidate.

## Auto-Filtering

The query search space is constrained using filters. An automatic filtering mechanism is implemented by which the application can automatically infer filters such as industries, location, and role preferences based on the professional summary. 

For this mechanism, the LLM is queried using a gated prompt mechanism which is validated using deterministic and subsequent LLM queries. Once an answer for one or more of the several questions about a user's profile passes a specification, they are excluded from subsequent LLM calls if the LLM is queried again to satisfy the unmet specifications. This expedites convergence of data extraction.

This is intended to only behave as a good starting point, and the user is expected to customize them to their needs.

## User Interface

The UI is built using `StreamLit`. It enables the user the input some text about themselves, as a professional summary, and auto-set the filters such as roles, industries, location, role type, and number of reranked results to fetch. Following this auto-inference or setting the filters manually (or not) the user may choose to perform the search.
The results of this search are shown one by one using the `Previous Role` and `Next Role` buttons. The button beneath this output box enables the user to apply to a selected role.

There are also some pre-defined examples, the same 27 professional summaries I used for examples. These can be accessed by the `Previous Example` and `Next Example` buttons on the top right, which will prefill the user input text box accordingly.

## Considerations

### Containerization

This project currently uses a containerized `Weaviate` server, but `Ollama` and the rest of the components are run locally. This is on purpose. During experimentation, running `Ollama` and the application un-containerized is an order of magnitude more performant than otherwise. The code to run `Ollama` as a container exists inside the `docker-compose.yml` as well as
an `entrypoint.sh` script to pull the required models inside the container. However, these components can be easily containerized if needed, into a 3-container setup: `Weaviate`, `Ollama`, and `Streamlit App`.

### DSPy

Parts of this project would directly benefit from some functionality offered by DSPy, as follows:

**Assertions**: When we are using LLM to extract data from a summary, we use multiple binary validations (whether the generation meets the spec), as found in `llm_utils.py:6-11`. This kind of functionality is where DSPy excels, while simultaneously improving the prompt using `teleprompters`/`optimizers` based on advanced Bayesian-Optimization algorithms like MIPRO.

## Data Flow sketch

![Weaviate](https://github.com/suprateembanerjee/Job-Search/assets/26841866/360601e8-9e06-40d9-9959-eefc99d1d64f)
