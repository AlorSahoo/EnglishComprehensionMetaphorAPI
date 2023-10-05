# EnglishComprehensionMetaphorAPI
A demo/prototype that uses cosine similarity, OpenAI API, and Metaphor API to determine if a user understood English articles about topics of their choice.

# Dependencies
OpenAI API, Metaphor API, numpy

# Functionality
The user submits a specific topic of their interest. Then, using Metaphor's neural search and the OpenAI API, we find and synthesize recent articles on the topic into a concise "reference" summary. The user is then provided the article links and asked to summarize them in 5 sentences. Then, using cosine similarity of the embeddings of the "reference" summary and "input" summary, we calculate how much the user understood the content that they read, in English.
