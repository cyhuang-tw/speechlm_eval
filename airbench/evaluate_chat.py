# -*- coding: utf-8 -*-
import json
import os
import random
import time
import logging
import shortuuid
from copy import deepcopy
from concurrent.futures.thread import ThreadPoolExecutor
import threading
import argparse
from func_timeout import func_set_timeout
import os
import time
import random
import requests
import jsonlines
from pathlib import Path
from openai import OpenAI

MAX_API_RETRY = 3
LLM_MIT_RETRY_SLEEP = 5

os.environ["MIT_SPIDER_TOKEN"] = ""
os.environ["MIT_SPIDER_URL"] = ""

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def load_file2list(path):
    res = []
    with open(path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            res.append(item)
    return res


# def mit_openai_api(**kwargs):
#     if not os.environ.get('MIT_SPIDER_TOKEN', None):
#         print("NO MIT_SPIDER_TOKEN FOUND, please set export MIT_SPIDER_TOKEN=<YOUR TOKEN>")
#     if not os.environ.get('MIT_SPIDER_URL', None):
#         print("NO MIT_SPIDER_URL FOUND, please set export MIT_SPIDER_URL=<YOUR URL>")
#     mit_spider_config = {
#         "url": os.environ.get("MIT_SPIDER_URL", None),
#         "header": {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {os.environ.get('MIT_SPIDER_TOKEN', None)}"
#         }
#     }
#     tenant = None
#     response = None
#     for i in range(MAX_API_RETRY):
#         try:
#             if tenant:
#                 payload = {'tenant': tenant}
#             else:
#                 payload = dict()
#             for k, w in kwargs.items():
#                 payload[f"{k}"] = w
#             response = requests.post(mit_spider_config['url'], json=payload, headers=mit_spider_config['header']).json()
#         except Exception as e:
#             print(response, e)
#             time.sleep(LLM_MIT_RETRY_SLEEP)
#             continue

#         if response['code'] == 200:
#             return response
#         else:
#             time.sleep(LLM_MIT_RETRY_SLEEP)
#             print(response)
#     return None


def mit_openai_api(**kwargs):
    for i in range(MAX_API_RETRY):
        try:
            response = client.chat.completions.create(**kwargs)
            return {
                "code": 200,
                "data": {
                    "response": {
                        "choices": [
                            {
                                "message": {
                                    "content": response.choices[0].message.content
                                },
                                "finish_reason": response.choices[0].finish_reason,
                            }
                        ]
                    },
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            }
        except Exception as e:
            print(e)
            time.sleep(LLM_MIT_RETRY_SLEEP)
    return None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
)
logger = logging.getLogger(__name__)
lock = threading.Lock()
finish_count = 0
failed_count = 0


@func_set_timeout(1200)
def get_result_by_request(**kwargs):
    response = mit_openai_api(**kwargs)
    if response["code"] == 200:
        result = response["data"]["response"]["choices"][0]["message"]["content"]
        prompt_tokens = response["data"]["prompt_tokens"]
        completion_tokens = response["data"]["completion_tokens"]
        finish_reason = response["data"]["response"]["choices"][0]["finish_reason"]
        return result, prompt_tokens, completion_tokens, finish_reason

    else:
        raise Exception(response["messages"])


def task(data, writer, args):
    global finish_count
    global failed_count
    openai_args = {
        "model": args.model_name,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    openai_args.update(data["openai_args"])
    for i in range(MAX_API_RETRY):
        try:
            result, prompt_tokens, completion_tokens, finish_reason = (
                get_result_by_request(**openai_args)
            )
            item = deepcopy(data)
            item.update(
                {
                    "gen": result,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "finish_reason": finish_reason,
                }
            )
            lock.acquire()
            finish_count += 1
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")
            writer.flush()
            lock.release()
            return

        except Exception as e:
            print("request error", e)
            pass

    lock.acquire()
    failed_count += 1
    lock.release()


def get_unprocessed_data(args, out_file):
    data_l = list()
    uuid_s = set()
    if os.path.exists(out_file):  # 以及生成到ouput里面的答案就不重新生成了
        out_data_l = load_file2list(out_file)
        for data in out_data_l:
            if data["gen"] != "error":
                uuid_s.add(data[args.uuid])
    for data in load_file2list(args.intermediate_file):
        if data[args.uuid] in uuid_s:
            continue
        data_l.append(data)
    return data_l


def run_chat_gen(args):
    if args.output_path:
        out_file = args.output_path
    else:
        out_file = os.path.splitext(args.jsonl_path)[0] + "_result.jsonl"
    items = get_unprocessed_data(args, out_file)

    pool = ThreadPoolExecutor(max_workers=args.num_workers)
    writer = open(out_file, "a", encoding="utf8")
    total_count = 0
    global finish_count, failed_count
    for item in items:
        total_count += 1
        pool.submit(task, item, writer, args)
    while finish_count + failed_count < total_count:
        logger.info(
            f"total:{total_count}  finish:{finish_count}  failed:{failed_count}"
        )
        time.sleep(10)

    time.sleep(10)
    writer.close()


def build_test_file(input_file, save_path):
    with open(save_path, "w", encoding="utf-8") as writer:
        with open(input_file, "r") as fp:
            for data in fp:
                row = json.loads(data)

                system_prompt = (
                    "You are a helpful and precise assistant for checking the quality of the answer.\n"
                    "[Detailed Audio Description]\nXAudioX\n[Question]\nXQuestionX\n"
                    "[The Start of Assistant 1s Answer]\nXAssistant1X\n[The End of Assistant 1s Answer]\n"
                    "[The Start of Assistant 2s Answer]\nXAssistant2X\n[The End of Assistant 2s Answer]\n[System]\n"
                    "We would like to request your feedback on the performance of two AI assistants in response to the user question "
                    "and audio description displayed above. AI assistants are provided with detailed audio descriptions and questions.\n"
                    "Please rate the helpfulness, relevance, accuracy, and comprehensiveness of their responses. "
                    "Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. "
                    "Please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. "
                    "The two scores are separated by a space."
                )

                path = row["path"]
                question = row["question"]
                answer_gt = row["answer_gt"]
                task_name = row["task_name"]
                dataset_name = row["dataset_name"]
                response = row["sllm_response"]

                if response == None:
                    continue

                if row.get("meta_info", None) == None:
                    print("lack meta info")
                    exit(1)
                else:
                    meta_info = row["meta_info"]

                content = (
                    system_prompt.replace("XAudioX", meta_info)
                    .replace("XQuestionX", question)
                    .replace("XAssistant1X", answer_gt)
                    .replace("XAssistant2X", response)
                )
                tmp_d = {
                    "uuid": shortuuid.uuid(),
                    "openai_args": {"messages": [{"role": "user", "content": content}]},
                    "meta_info": meta_info,
                    "path": path,
                    "question": question,
                    "answer_gt": answer_gt,
                    "task_name": task_name,
                    "dataset_name": dataset_name,
                    "Audio-LLM-response": response,
                }
                if random.random() < 0.5:
                    tmp_d["openai_args"].update({"temperature": 2.0})
                writer.write(json.dumps(tmp_d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm gen")
    parser.add_argument("--jsonl_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--intermediate_file", type=Path, default="batch_run_input.jsonl"
    )
    parser.add_argument("-n", "--num-workers", type=int, default=50)  # max=50
    parser.add_argument("-m", "--model-name", type=str, default="gpt-4-0125-preview")
    parser.add_argument("-t", "--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--uuid", type=str, default="uuid")

    args = parser.parse_args()

    # step 1
    build_test_file(args.jsonl_path, args.intermediate_file)

    # step 2
    run_chat_gen(args)
