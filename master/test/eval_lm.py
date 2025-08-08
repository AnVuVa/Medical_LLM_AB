import argparse
from ..rag_pipeline import qa_prompt
from ..rag_pipeline import ChatAssistant
from ..utils import load_qa_dataset, load_prepared_retrieve_docs

from typing import List, Optional
from langchain.schema import Document

def get_answer_from_response(llm_response: str) -> str:
    return llm_response.strip()

def build_qa_prompt(question: str, document: Optional[List[Document]]) -> str:
    if document is not None:
        document = '\n'.join([f"Document {i+1}:\n" + doc.page_content for i,doc in enumerate(document)])
    
    return qa_prompt.format(question=question, document=document)

def process_question(question, prompt, answer, id, args, llm):
    llm_response = llm.get_response("", prompt)
    # ans = get_answer_from_response(llm_response)
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"ID: {id}\n")
        f.write(prompt)
        f.write(f"LLM Response:\n{llm_response}\n")
        f.write(f"Answer: {answer} \n\n")

    # with open("log_score.txt", "a", encoding="utf-8") as f:
    #     f.write("1" if ans == answer else "0")
    # return 1 if ans == answer else 0
    return llm_response

def evaluate_qa(questions, prompts, answers, ids, args, llm):
    import concurrent.futures
    from tqdm import tqdm
    ans = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_question, questions[i], prompts[i], answers[i], ids[i], args, llm) for i in range(len(questions))]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(questions)):
            ans.append(future.result())
    return ans

def main(args):
    ids, questions, options, answers = load_qa_dataset(args.qa_file)

    if ids is None:
        raise ValueError(f"No id field in {args.qa_file}.")
    
    if args.num_docs > 0:
        if args.prepared_retrieve_docs_path is not None:
            documents = load_prepared_retrieve_docs(args.prepared_retrieve_docs_path)
            docs = [d[:args.num_docs] for i,d in enumerate(documents)]
        else:
            raise ValueError(f"No prepared retrieve docs found.")
    else:
        docs = [None]*len(questions)

    prompts = [build_qa_prompt(questions[i], docs[i]) for i in range(len(questions))]

    llm = ChatAssistant(args.model_name, args.provider)

    with open("log_score.txt", "a", encoding="utf-8") as f:
            f.write("\n")

    qa_results = evaluate_qa(questions, prompts, answers, ids, args, llm)
    qa_results = [qa_results[i][qa_results[i].rfind("[")+1:qa_results[i].rfind("]")] for i in range(len(qa_results))]
    # print(f"{qa_results}")
    import pyperclip
    pyperclip.copy('\n'.join(qa_results))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--qa_file", type=str, default="dataset/QA Data/random.jsonl")
    parser.add_argument("--prepared_retrieve_docs_path", type=str, default="prepared_retrieve_docs.pkl")

    parser.add_argument("--model_name", type=str, default="mistral-medium")
    parser.add_argument("--provider", type=str, default="mistral")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--num_docs", type=int, default=0)

    parser.add_argument("--dataset_path", type=str)

    args = parser.parse_args()

    print(args)

    main(args)