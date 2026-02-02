import boto3
import os
from dotenv import load_dotenv

load_dotenv()
REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")

client = boto3.client('bedrock', region_name=REGION)

try:
    response = client.list_foundation_models()
    with open("models.txt", "w", encoding="utf-8") as f:
        f.write(f"Models in {REGION}:\n")
        for model in response['modelSummaries']:
            if "text" in model['modelId'] or "claude" in model['modelId']:
                 f.write(f"- {model['modelId']}\n")
                 print(f"- {model['modelId']}")
except Exception as e:
    print(f"Error listing models: {e}")
