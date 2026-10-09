### COMPLETE THE CODE ###

from rag_tool import build_retriever
from rag import answer_question


def main():
    # Build the retriever over the single-document knowledge base.
    retriever = build_retriever(k=4)

    # The graded question: the IT Service Desk phone number.
    question = "What exact phone number should I call for the university IT Service Desk?"

    answer = answer_question(question, retriever)

    print("QUESTION:", question)
    print()
    print("ANSWER:", answer)


if __name__ == "__main__":
    main()
