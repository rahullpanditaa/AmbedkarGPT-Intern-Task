from retrieval.retrieval_logic import Search


def main():
    searcher = Search()
    print(searcher.build_vector_db())


if __name__ == "__main__":
    main()
