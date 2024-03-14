from flask import Flask

from smartassignments import *

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, World!"

@app.route("/getSimilarity")
def smart_assignments(embeddings, task):
    # Compare similarity between embeddings and task
    # Return the similarity score
    taskEmbeddings = GetEmbeddings(client, task)

    similarity_score = GetSimilarity(embeddings, taskEmbeddings)

    return similarity_score

@app.route("/getPromptResponse")
def get_smart_assignment_prompt_response(prompt, max_tokens):
    return GetPromptResponse(client, prompt, max_tokens)
    
if __name__ == "__main__":
    client = GetOpenAiClient()
    app.run()

