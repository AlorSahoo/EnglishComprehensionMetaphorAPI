import openai
from metaphor_python import Metaphor
import numpy as np
from numpy.linalg import norm

openai.api_key = "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"

metaphor = Metaphor(api_key="c48dc95c-d30f-4d5a-9337-d3e9867e4f87")

topic = input("Tell me a very specific topic you are interested in (ex: potato farming practices): ")

USER_QUESTION = "What's the recent news on " + topic + "?"

SYSTEM_MESSAGE_QUERY = "You are a helpful assistant that generates search queries based on user questions. Only generate one search query."

query_cleanup = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYSTEM_MESSAGE_QUERY},
        {"role": "user", "content": USER_QUESTION},
    ],
)
query = query_cleanup.choices[0].message.content
search_response = metaphor.search(
    query, use_autoprompt=True, start_published_date="2023-06-01"
)

#Get topic IDs
topicIDs = []
unedited_output = "These are the articles in order: "
urls = []
for i in range(min(3, len(search_response.results))):

    #get ID
    id = search_response.results[i].id
    #add URLs to link
    urls.append(search_response.results[i].url)
    #extract content from articles
    response = metaphor.get_contents([id,]).contents[0].extract

    #concatenate into string for future summary
    dividing_text = "\nThis is article number " + str(i) + ": \n"
    dividing_text += response
    unedited_output += dividing_text

#Clean up the response

SYSTEM_MESSAGE_CLEANUP = "You are editing assistant. You will receive an input of text containing multiple articles. " \
                         "Remove irrelevant links and words. Synthesize the articles into 1 cohesive 5 sentence summary."

completion_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYSTEM_MESSAGE_CLEANUP},
        {"role": "user", "content": unedited_output},
    ],
)

#Provide elevant articles
print('Read these articles: ', urls)

#Ask for user summary
user_summary = input('Summarize them in a 5 sentence summary: ')

#summarized reference
summarized_output = completion_response.choices[0].message.content

#Get embeddings
response_reference = openai.Embedding.create(input=[summarized_output], engine="text-embedding-ada-002")
response_user_input = openai.Embedding.create(input=[user_summary], engine="text-embedding-ada-002")

response_embedding = np.array(response_reference['data'][0]['embedding'])
response_user_input = np.array(response_user_input['data'][0]['embedding'])

#calculate cosine similarity
cosine_similarity = np.dot(response_embedding,response_user_input)/(norm(response_embedding)*norm(response_user_input))

print('Here is an index representing how good your summary was: ', cosine_similarity)

print("Here is the reference summary: ", summarized_output)

#Testing reveals that similarity scores of < 0.5 correlate with summaries that are largely irrelevant; scores of 0.9
#or higher demonstrate proficiency in the content



