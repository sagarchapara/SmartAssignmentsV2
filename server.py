from flask import Flask, jsonify
import json

from smartassignments import *
from vectordb import *
from flask import request

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, World!" 

@app.route("/relevantAssignments", methods=["POST"])
def smart_assignments():
    data = request.get_json()
    resources = data["resourceAadIds"]
    task = str(data["taskData"])

    # Get embeddings for resources
    collection_data = Get(collection, resources)
    embeddings = collection_data["embeddings"]

    # Compare similarity between embeddings and task
    # Return the similarity score
    taskEmbeddings = GetEmbeddings(openAIClient, task)

    similarity_score = GetSimilarity(embeddings, taskEmbeddings)

    return jsonify(np.array(similarity_score).tolist())

@app.route("/prompt", methods=["POST"])
def get_smart_assignment_prompt_response():
    data = request.get_json()
    prompt = data["prompt"]
    
    response = GetPromptResponse(openAIClient, prompt)

    return jsonify(response)
    
if __name__ == "__main__":
    openAIClient = GetOpenAiClient()
    chromaCollection = CreateCollection("user-store")
    app.run()

