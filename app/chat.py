import os, sys
# Fix import path so it works with python app/chat.py
sys.path.append(os.path.dirname(__file__))

from search import search, ask_llm

print("🧠 Sangam-Brain Ready! Ask anything from your September PDF.")
print("Type 'exit' to quit.\n")

while True:
    try:
        q = input("You: ")
        if q.lower().strip() in ["exit", "quit", "q"]:
            print("Bye! 👋")
            break
        if not q.strip():
            continue
        
        ctx = search(q, top_k=4)  # increased to 4 for better answers
        ask_llm(q, ctx)
        print("\n" + "-"*60 + "\n")
    except KeyboardInterrupt:
        print("\nBye! 👋")
        break