#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""模仿道德经创作，并提供译文和讨论道德经相关内容。

使用前请将 OPENAI_API_KEY 环境变量设成您的 openAI API key 值：

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


defaultdict = collections.defaultdict

LUO_DAYOU_LYRICS_FILE = "data/data.txt"
NUM_CHARS_PER_SONG = 200


def ParseSongs(path: str) -> Dict[str, List[str]]:
    """Parses the given lyrics file.

    Returns:
        Map from song name to lines of lyrics.
    """

    songs: Dict[str, List[str]] = defaultdict(list)
    song = None
    for line in io.open(path, mode="r", encoding="utf-8").readlines():
        line = line.strip()
        if line.startswith("第"): # if it's title just ignore
            title = line
            song = songs[title]
            continue

        if not line:
            continue
        prefix = ""
        for ch in line:
            if ch == "（" or ch=="第" or ch =="《" :  # ignore all characters after "(" and all chapter name
                break
            if ord(ch) < 128 or ch in "　！…、—○《》":  # Treat non-Chinese as separater.
                if prefix:
                    # assert song is not None, "Error: song subject cannot be None"
                    song.append(prefix)
                    prefix = ""
            else:
                prefix += ch
        if prefix:
            # assert song is not None, 
            song.append(prefix)
            prefix = ""
    return songs


def NormalizeFileLines(path: str) -> List[str]:
    lines: List[str] = []
    for song_lines in ParseSongs(path).values():
        lines.extend(song_lines)
    return lines


def BuildUnigramFrequencyMap(lines: List[str]) -> Dict[str, float]:
    map: Dict[str, float] = defaultdict(lambda: 0)
    for line in lines:
        for ch in line:
            map[ch] += 1
        map[""] += 1
    return map


def BuildBigramFrequencyMap(lines: List[str]) -> Dict[str, Dict[str, float]]:
    map: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0))
    for line in lines:
        ch0 = ""
        ch1 = ""
        for ch1 in line:
            map[ch0][ch1] += 1
            ch0 = ch1
        map[ch1][""] += 1
    return map


def BuildTrigramFrequencyMap(
    lines: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    map: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 0))
    )
    for line in lines:
        ch0 = ""
        ch1 = ""
        ch2 = ""
        for ch2 in line:
            map[ch0][ch1][ch2] += 1
            ch0 = ch1
            ch1 = ch2
        map[ch1][ch2][""] += 1
        map[ch2][""][""] += 1
    return map


def BuildQuadgramFrequencyMap(
    lines: List[str],
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    map: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    )
    for line in lines:
        ch0 = ""
        ch1 = ""
        ch2 = ""
        ch3 = ""
        for ch3 in line:
            map[ch0][ch1][ch2][ch3] += 1
            ch0 = ch1
            ch1 = ch2
            ch2 = ch3
        map[ch1][ch2][ch3][""] += 1
        map[ch2][ch3][""][""] += 1
        map[ch3][""][""][""] += 1
    return map


def BuildPantaFrequencyMap(
    lines: List[str],
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]:
    map: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = defaultdict(lambda:defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))) )
    )
    for line in lines:
        ch0 = ""
        ch1 = ""
        ch2 = ""
        ch3 = ""
        ch4 = ""
        for ch4 in line:
            map[ch0][ch1][ch2][ch3][ch4] += 1
            ch0 = ch1
            ch1 = ch2
            ch2 = ch3
            ch3 = ch4
        map[ch1][ch2][ch3][ch4][""] += 1
        map[ch2][ch3][ch4][""][""] += 1
        map[ch3][ch4][""][""][""] += 1
        map[ch4][""][""][""][""]+=1
    return map

def BuildSixthFrequencyMap(
    lines: List[str],
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]]:
    map: Dict[str, [str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]] = defaultdict(lambda: defaultdict(lambda:defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))) )
    )
    for line in lines:
        ch0 = ""
        ch1 = ""
        ch2 = ""
        ch3 = ""
        ch4 = ""
        ch5 = ""
        for ch5 in line:
            map[ch0][ch1][ch2][ch3][ch4][ch5] += 1
            ch0 = ch1
            ch1 = ch2
            ch2 = ch3
            ch3 = ch4
            ch4 = ch5
        map[ch1][ch2][ch3][ch4][ch5][""] += 1
        map[ch2][ch3][ch4][ch5][""][""] += 1
        map[ch3][ch4][ch5][""][""][""] += 1
        map[ch4][ch5][""][""][""] [""]+= 1
        map[ch5][""][""][""][""][""]+=1
    return map


def PrintFrequencyMap(freq_map: Dict[str, float]) -> None:
    freq_list: List[Tuple[float, str]] = []
    for ch, count in freq_map.items():
        freq_list.append((count, ch))
    freq_list = sorted(freq_list, reverse=True)

    for count, ch in freq_list:
        print("%s: %d" % (ch, count))


T = TypeVar("T")


def PickTopP(freq_map: Dict[T, float], top_p: float) -> Dict[T, float]:
    """Returns the prefix with the given probability mass."""

    assert 0 <= top_p
    assert top_p <= 1

    sorted_list = sorted(
        freq_map.items(), key=lambda ch_and_freq: ch_and_freq[1], reverse=True
    )
    total_freq = sum(freq_map.values())
    threshold = total_freq * top_p
    answer: Dict[T, float] = {}
    accumulated_freq = 0
    for x, freq in sorted_list:
        answer[x] = freq
        accumulated_freq += freq
        if accumulated_freq >= threshold:
            return answer
    return answer


def AdjustWeightByTemperature(
    freq_map: Dict[str, float], temperature: float
) -> Dict[str, float]:

    assert 0 <= temperature
    assert temperature <= 2

    if temperature < 0.01:
        temperature = 0.01  # Avoid overflow with tiny temperature value.
    total_freq = sum(freq_map.values())
    return {
        ch: math.pow(freq / total_freq, 1 / temperature)
        for ch, freq in freq_map.items()
    }


def WeightedSample(freq_map: Dict[str, float]) -> str:
    """Picks one char randomly."""

    total_weight: float = sum(freq_map.values())
    # random() generates a random float in [0, 1).
    r = random.random() * total_weight
    start = 0
    for x, weight in freq_map.items():
        if start <= r and r < start + weight:
            return x
        start += weight
    return ""


def WeightedSampleWithTemperature(
    freq_map: Dict[str, float], temperature: float
) -> str:
    """Picks one char randomly.

    Args:
        freq_map: map from a char to its frequency
        temperature: a float in [0, 2] that determines how wild the pick can be.
            0 means that we will always pick the char with the highest frequency.
            1 means that the probability of a char being picked is proportional
            to its frequency in the map.
    """

    assert 0 <= temperature
    assert temperature <= 2

    return WeightedSample(AdjustWeightByTemperature(freq_map, temperature))


def WeightedSampleWithTemperatureTopP(
    freq_map: Dict[str, float], temperature: float, top_p: float
) -> str:
    """Picks one char randomly.

    Args:
        freq_map: map from a char to its frequency
        temperature: a float in [0, 1] that determines how wild the pick can be.
            0 means that we will always pick the char with the highest frequency.
            0.5 means that the probability of a char being picked is proportional
            to its frequency in the map.
            1 means that all chars in `freq_map` are considered with equal probability.
        top_p: a float in [0, 1] that determines how tolerant the pick is.
            0 means that only the best choice is considered.
            0.5 means the best half of the valid choices are considered.
            1 means that all valid choices are considered.
    """

    assert 0 <= temperature
    assert temperature <= 2
    assert 0 <= top_p
    assert top_p <= 1

    return WeightedSampleWithTemperature(
        PickTopP(freq_map, top_p), temperature=temperature
    )


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


def GetChar(text: str, index: int) -> str:
    """Get the char at the given index, or "" if the index is invalid."""

    try:
        return text[index]
    except IndexError:
        return ""


def GetApiKey() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    print(api_key)
    if not api_key:
        sys.exit("请首先将 OPENAI_API_KEY 环境变量设成您的 openAI API key。")
    return api_key


def GetSampleSong() -> Tuple[str, List[str]]:
    """Gets a sample song written by 罗大佑, based on the user selection.

    Returns:
        a (song title, lyrics) pair
    """

    songs = ParseSongs(LUO_DAYOU_LYRICS_FILE)
    titles = sorted(songs.keys())
    for i, title in enumerate(titles):
        print(f"{i}. {title}")
    index = input(f"请选择讨论哪一章节 (0-{len(titles) - 1}): ")
    index = int(index)
    title = titles[index]
    return title, songs[title]


def MakeMessages(
    hint: str, conversation: List[Tuple[str, str]], new_ask: str
) -> List[Dict[str, str]]:
    """Generates messages to send to chatGPT from the conversation history."""

    # Messages in the reverse order.
    reverse_messages = [
        {
            "role": "user",
            "content": new_ask,
        },
    ]
    total_length = 0
    for ask, answer in reversed(conversation):
        total_length += len(ask) + len(answer)
        # If the total conversion is too long, keep the suffix only.
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


def GenerateLyricsByChatGpt(subject: str, temperature: float, top_p: float) -> None:
    """Generates lyrics using the chatGPT model."""

    api_key = GetApiKey()
    sample_title, sample_lyrics = GetSampleSong()
    print(sample_title)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    hint = "你是一个文学修养高深的古文作者。你阅读过道德经并理解它的意思。"
    ask1 = f"查看“{sample_title}”这一章节的原文"
    answer1 = "\n".join(sample_lyrics)
    conversation = [(ask1, answer1)]
    new_ask = f"根据这章的主题写一首古文，模仿道德经的风格，比如前面这首"

    print(f"给chatGPT的提示：\n{hint}\n")
    print(f"您：\n{ask1}\n")
    print(f"chatGPT的回答：\n{answer1}\n")
    print(f"您：\n{new_ask}\n")

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
        print(f"chatGPT的回答：\n{answer}\n")

        conversation.append((new_ask, answer))
        new_ask = input("请输入您的意见，或提出其他问题：")


def main():
    # Parse the flags.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--chatgpt",
        help="用 chatGPT 模型来产生歌词",
        default=False,
        action="store_true",
    )
    # parser.add_argument(
    #     "-d",
    #     "--davinci",
    #     help="用 Davinci 模型来产生歌词",
    #     default=False,
    #     action="store_true",
    # )
    parser.add_argument(
        "-t",
        "--temperature",
        help="想象力有多狂野 ([0, 2]区间的实数)",
        type=FloatFrom0To2,
        default=0.8,
    )
    parser.add_argument(
        "-p",
        "--top_p",
        help="只考虑头部概率的选择 ([0, 1]区间的实数)",
        type=FloatFrom0To1,
        default=1,
    )
    parser.add_argument(
        "subject",
        nargs="?",
        help="歌曲的主题",
        default="",
    )
    args = parser.parse_args()

    if args.chatgpt:
        GenerateLyricsByChatGpt(
            args.subject, temperature=args.temperature, top_p=args.top_p
        )
        return

    if args.davinci:
        GenerateLyricsByDavinci(
            args.subject, temperature=args.temperature, top_p=args.top_p
        )
        return

    random.seed()
    lines = NormalizeFileLines(LUO_DAYOU_LYRICS_FILE)
    uni_freq_map = BuildUnigramFrequencyMap(lines)
    bi_freq_map = BuildBigramFrequencyMap(lines)
    tri_freq_map = BuildTrigramFrequencyMap(lines)
    quad_freq_map = BuildQuadgramFrequencyMap(lines)
    panta_freq_map = BuildPantaFrequencyMap(lines)
    sixth_freq_map = BuildSixthFrequencyMap(lines)

    lyrics = args.subject
    for _ in range(NUM_CHARS_PER_SONG):
        ch = WeightedSampleWithTemperatureTopP(
            uni_freq_map, temperature=args.temperature, top_p=args.top_p
        )
        if ch:
            lyrics += ch
        else:
            lyrics += "\n"
    print("一阶----")
    print(lyrics)

    lyrics = args.subject
    ch = GetChar(lyrics, -1)
    for _ in range(NUM_CHARS_PER_SONG):
        freq_map: Dict[str, float] = bi_freq_map[ch]
        ch = WeightedSampleWithTemperatureTopP(
            freq_map, temperature=args.temperature, top_p=args.top_p
        )
        if ch:
            lyrics += ch
        else:
            lyrics += "\n"
    print("二阶----")
    print(lyrics)

    lyrics = args.subject
    ch0 = GetChar(lyrics, -2)
    ch1 = GetChar(lyrics, -1)
    for _ in range(NUM_CHARS_PER_SONG):
        freq_map: Dict[str, float] = tri_freq_map[ch0][ch1]
        if len(freq_map) <= 1:
            freq_map = bi_freq_map[ch1]
        ch2 = WeightedSampleWithTemperatureTopP(
            freq_map, temperature=args.temperature, top_p=args.top_p
        )
        if ch2:
            lyrics += ch2
        else:
            lyrics += "\n"
        ch0 = ch1
        ch1 = ch2
    print("三阶----")
    print(lyrics)

    lyrics = args.subject
    ch0 = GetChar(lyrics, -3)
    ch1 = GetChar(lyrics, -2)
    ch2 = GetChar(lyrics, -1)
    for _ in range(NUM_CHARS_PER_SONG):
        freq_map: Dict[str, float] = quad_freq_map[ch0][ch1][ch2]
        if len(freq_map) <= 1:
            freq_map = tri_freq_map[ch1][ch2]
        if len(freq_map) <= 1:
            freq_map = bi_freq_map[ch2]
        ch3 = WeightedSampleWithTemperatureTopP(
            freq_map, temperature=args.temperature, top_p=args.top_p
        )
        if ch3:
            lyrics += ch3
        else:
            lyrics += "\n"
        ch0 = ch1
        ch1 = ch2
        ch2 = ch3
    print("四阶----")
    print(lyrics)

    lyrics = args.subject
    ch0 = GetChar(lyrics, -4)
    ch1 = GetChar(lyrics, -3)
    ch2 = GetChar(lyrics, -2)
    ch3 = GetChar(lyrics, -1)
    for _ in range(NUM_CHARS_PER_SONG):
        freq_map: Dict[str, float] = panta_freq_map[ch0][ch1][ch2][ch3]
        if len(freq_map) <= 1:
            freq_map = quad_freq_map[ch1][ch2][ch3]
        if len(freq_map) <= 1:
            freq_map = tri_freq_map[ch2][ch3]
        if len(freq_map) <= 1:
            freq_map = bi_freq_map[ch3]
        ch4 = WeightedSampleWithTemperatureTopP(
            freq_map, temperature=args.temperature, top_p=args.top_p
        )
        if ch4:
            lyrics += ch4
        else:
            lyrics += "\n"
        ch0 = ch1
        ch1 = ch2
        ch2 = ch3
        ch3 = ch4
    print("五阶----")
    print(lyrics)

    lyrics = args.subject
    ch0 = GetChar(lyrics, -5)
    ch1 = GetChar(lyrics, -4)
    ch2 = GetChar(lyrics, -3)
    ch3 = GetChar(lyrics, -2)
    ch4 = GetChar(lyrics, -1)
    for _ in range(NUM_CHARS_PER_SONG):
        freq_map: Dict[str, float] = sixth_freq_map[ch0][ch1][ch2][ch3][ch4]
        if len(freq_map) <= 1:
            freq_map = panta_freq_map[ch1][ch2][ch3][ch4]
        if len(freq_map) <= 1:
            freq_map = quad_freq_map[ch2][ch3][ch4]
        if len(freq_map) <= 1:
            freq_map = tri_freq_map[ch3][ch4]
        if len(freq_map) <= 1:
            freq_map = bi_freq_map[ch4]
        ch5 = WeightedSampleWithTemperatureTopP(
            freq_map, temperature=args.temperature, top_p=args.top_p
        )
        if ch5:
            lyrics += ch5
        else:
            lyrics += "\n"
        ch0 = ch1
        ch1 = ch2
        ch2 = ch3
        ch3 = ch4
        ch4 = ch5
    print("六阶----")
    print(lyrics)


if __name__ == "__main__":
    main()
