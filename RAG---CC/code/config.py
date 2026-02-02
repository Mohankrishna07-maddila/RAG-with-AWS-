import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID")
MODEL_ARN = os.getenv("MODEL_ARN")

print("AWS_REGION:", AWS_REGION)
print("KNOWLEDGE_BASE_ID:", "SET" if KNOWLEDGE_BASE_ID else "MISSING")
print("MODEL_ARN:", MODEL_ARN if MODEL_ARN else "MISSING")

if not KNOWLEDGE_BASE_ID or not MODEL_ARN:
    raise RuntimeError(
        "Missing required configuration. "
        "Check KNOWLEDGE_BASE_ID and MODEL_ARN in .env"
    )
