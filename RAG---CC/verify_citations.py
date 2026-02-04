from gradio_app import retrieve_context, generate_answer, ask_hybrid_rag
import sys

# Test query
query = "What is cloud computing?"

print(f"Testing RAG Citations with query: '{query}'")

try:
    # 1. Test retrieve_context
    context, sources = retrieve_context(query)
    print(f"\n--- Sources Found ---")
    print(sources)
    print("---------------------")
    
    if not sources:
        print("WARNING: No sources found. (This might be normal if docs aren't indexed yet, or config is wrong)")
    else:
        print("PASS: Sources extracted.")

    # 2. Test full flow
    answer, context, sources = ask_hybrid_rag(query)
    print("\n--- Final Answer ---")
    print(answer)
    print("--------------------")
    
    if "**Sources:**" in answer:
        print("PASS: Citations detected in answer.")
    else:
        print("WARNING: No citations in answer (maybe no sources found).")

except Exception as e:
    print(f"TEST CRASHED: {e}")
    sys.exit(1)
