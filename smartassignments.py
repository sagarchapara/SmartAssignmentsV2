import os
from openai import AzureOpenAI
    
client = AzureOpenAI(
    api_key=os.getenv("28f146892bb54fd3a8ed90c8f7080f0a"),  
    azure_endpoint=os.getenv("https://assignments.openai.azure.com/")
)