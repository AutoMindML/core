from automind.process.ollama import run_ollama

if __name__ == "__main__":
    print("query ollama...")

    buffer_list = run_ollama("generate 50 line python code.")

    for line in buffer_list:
        print(line, end="")
