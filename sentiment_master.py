#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""使用前请将 OPENAI_API_KEY 环境变量设成您的 openAI API key 值：

export OPENAI_API_KEY=<您的 openAI API key>
"""

from typing import Dict, List, Tuple, TypeVar

import argparse
import collections
import io
import json
import math
import os
import random
import requests
import sys

def FloatFrom0To2(text: str) -> float:
    try:
        x = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text} is not a floating-point literal.")

    if x < 0.0 or x > 2.0:
        raise argparse.ArgumentTypeError(f"{text} is not in the range [0.0, 2.0].")
    return x


def FloatFrom0To1(text: str) -> float:
    try:
        x = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text} is not a floating-point literal.")

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{text} is not in the range [0.0, 1.0].")
    return x




def GetApiKey() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    print(api_key)
    if not api_key:
        sys.exit("请首先将 OPENAI_API_KEY 环境变量设成您的 openAI API key。")
    return api_key




def MakeMessages(
    hint: str, conversation: List[Tuple[str, str]], new_ask: str
) -> List[Dict[str, str]]:
    """Generates messages to send to chatGPT from the conversation history."""

    reverse_messages = [
        {
            "role": "user",
            "content": new_ask,
        },
    ]
    total_length = 0
    for ask, answer in reversed(conversation):
        total_length += len(ask) + len(answer)
        if total_length > 4096:
            break

        reverse_messages.append(
            {
                "role": "assistant",
                "content": answer,
            },
        )
        reverse_messages.append(
            {
                "role": "user",
                "content": ask,
            },
        )
    reverse_messages.append(
        {
            "role": "system",
            "content": hint,
        },
    )

    return list(reversed(reverse_messages))


def GenerateLyricsByChatGpt(temperature: float, top_p: float) -> None:
    """Generates lyrics using the chatGPT model."""

    api_key = GetApiKey()
  

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    hint = "you just read what I send, it should be a article, what is the the sentiment of this article, your answer should have two parts, first is the answer of the sentiment question and your analyse, the second just be a number called output, if it's positive, return 1, if it's negative, return 0, if you don't know or it's mixed, return 2"

    conversation = []
    new_ask = input("lease provide the statement you would like me to analyze the sentiment of：")

 

    completion_endpoint = "https://api.openai.com/v1/chat/completions"
    while True:
        messages = MakeMessages(hint, conversation, new_ask)
        data = json.dumps(
            {
                "model": "gpt-3.5-turbo-0301",
                "messages": messages,
                "max_tokens": 1024,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": 2,
                "presence_penalty": 1,
            }
        )

        result = requests.post(completion_endpoint, headers=headers, data=data)
        answer = result.json()["choices"][0]["message"]["content"]
        print()
        print(f"Answer：\n{answer}\n")

        conversation.append((new_ask, answer))
        new_ask = input("Please provide the statement you would like me to analyze the sentiment of：：")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--chatgpt",
        help="用 chatGPT 模型来产生歌词",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=FloatFrom0To2,
        default=0.8,
    )
    parser.add_argument(
        "-p",
        "--top_p",
        type=FloatFrom0To1,
        default=1,
    )

    args = parser.parse_args()

    if args.chatgpt:
        GenerateLyricsByChatGpt(
           temperature=args.temperature, top_p=args.top_p
        )
        return

if __name__ == "__main__":
    main()
