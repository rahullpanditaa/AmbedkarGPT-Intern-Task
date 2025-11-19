from lib.rag_chain import create_rag_chain


def main():
    chain = create_rag_chain()
    print("Welcome to AmbedkarGPT!!")
    print("Starting REPL...")
    while True:
        question = input("> ")
        if question.lower() == "exit":
            break
        response = chain.invoke(question)
        print(response)


if __name__ == "__main__":
    main()
