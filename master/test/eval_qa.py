import argparse
from ..rag_pipeline import multichoice_qa_prompt
from ..rag_pipeline import ChatAssistant
from ..utils import paralelize, load_qa_dataset, load_prepared_retrieve_docs

from datetime import datetime
from typing import List, Optional
from langchain.schema import Document

def get_answer_from_response(llm_response: str) -> chr:
    """
    Get the answer from the LLM response.
    """
    return llm_response[llm_response.lower().rfind("the answer is ") + 14]

def build_multichoice_qa_prompt(question: str, options: str, document: Optional[List[Document]]) -> str:
    """
    Build the prompt for the multichoice QA task.
    """
    if document is not None:
        document = '\n'.join([f"Document {i+1}:\n" + doc.page_content for i,doc in enumerate(document)])
    
    return multichoice_qa_prompt.format(question=question, options=options, document=document)

def process_question(question, prompt, answer, id, args, llm):
    llm_response = ""
    for j in range(args.retries):
        try:
            llm_response = llm.get_response("", prompt)
            ans = get_answer_from_response(llm_response)
            if ans in ["A", "B", "C", "D", "E"]:
                with open("log.txt", "a", encoding="utf-8") as f:
                    f.write(f"ID: {id}\n")
                    f.write(prompt)
                    f.write(f"LLM Response:\n{llm_response}\n")
                    f.write(f"Answer: {answer}  {ans}\n\n")
                break
        except Exception as e:
            print(f"Error: {e}")
            ans = "#"
    with open("log_score.txt", "a", encoding="utf-8") as f:
        f.write("1" if ans == answer else "0")
    return 1 if ans == answer else 0

def evaluate_qa(questions, prompts, answers, ids, args, llm):
    import concurrent.futures
    from tqdm import tqdm
    correct = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_question, questions[i], prompts[i], answers[i], ids[i], args, llm) for i in range(len(questions))]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(questions)):
            correct += future.result()
    return correct / len(questions)


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

    prompts = [build_multichoice_qa_prompt(questions[i], options[i], docs[i]) for i in range(len(questions))]

    # print(prompts[0])
    llm = ChatAssistant(args.model_name, args.provider)

    with open("log_score.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{datetime.now()} {args}\n")

    acc = evaluate_qa(questions, prompts, answers, ids, args, llm)
    print(f"Accuracy: {acc}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument("--qa_file", type=str, default="dataset/QA Data/MedAB/MedABv2.jsonl")
    # parser.add_argument("--prepared_retrieve_docs_path", type=str, default="dataset/QA Data/MedAB/prepared_retrieve_docs_full.pkl")

    parser.add_argument("--qa_file", type=str, default="dataset/QA Data/MedMCQA/translated_hard_questions.jsonl")
    parser.add_argument("--prepared_retrieve_docs_path", type=str, default="dataset/QA Data/MedMCQA/prepared_retrieve_docs_full.pkl")

    # Eval params
    parser.add_argument("--model_name", type=str, default="mistral-medium")
    parser.add_argument("--provider", type=str, default="mistral")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--num_docs", type=int, default=0)
    parser.add_argument("--retries", type=int, default=4)


    # Dataset params
    parser.add_argument("--dataset_path", type=str)

    args = parser.parse_args()
    print(f"Log:{args}")

    main(args)