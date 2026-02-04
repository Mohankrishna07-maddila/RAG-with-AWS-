from gradio_app import ask_hybrid_rag
import sys

# Test query
query = "What are the main topics in unit 3 of the cloud computing course?"

print(f"Testing RAG with query: '{query}'")

try:
    response, context, sources = ask_hybrid_rag(query)
    print("\n--- Response from Bedrock ---")
    print(response)
    print("-----------------------------")
    
    with open("verification_result.txt", "w") as f:
        f.write(response)

    if "Error" in response:
        print("TEST FAILED: Error detected in response.")
        sys.exit(1)
    else:
        print("TEST PASSED: execution successful.")
        
except Exception as e:
    print(f"TEST CRASHED: {e}")
    sys.exit(1)
