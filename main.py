from retrieval.rag_chain import create_rag_chain


def main():
    while True:
        question = input("> ")
        if question.lower() == "exit":
            break
        response = create_rag_chain().invoke(question)
        print(response)


if __name__ == "__main__":
    main()
