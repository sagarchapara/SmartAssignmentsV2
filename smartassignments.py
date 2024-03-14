import os
from openai import AzureOpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def ReadAndGenerateEmbeddings(client, fileName):
    #read from text file
    with open(fileName, 'r') as file:
        data = file.read().replace('\n', ',')

    return GetEmbeddings(client, data)

def GetEmbeddings(client, data):
    # data = clean_text(data)   
    return GenerateEmbedding(client, data)

def GenerateEmbedding(client, data):
    embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input= data
    )

    #data is a list of embeddings
    #can you return 2d array of np arrays
    embeddings = [np.array(embedding.data[i].embedding) for i in range(len(embedding.data))]
    return np.array(embeddings)

def GetSimilarity(embeddings1, embeddings2):
    similarity = cosine_similarity(embeddings1, embeddings2)
    return similarity

def GetPromptResponse(client, messages):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
            {"role": "user", "content": "Who were the founders of Microsoft?"}
        ]
    )
    return response.choices[0].message.content


def GetOpenAiClient():
    return AzureOpenAI(
        api_key= "28f146892bb54fd3a8ed90c8f7080f0a",
        api_version="2023-12-01-preview",
        azure_endpoint= "https://assignments.openai.azure.com/")
