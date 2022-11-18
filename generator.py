
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
import re
from tqdm import tqdm
import json
import os
import pandas as pd
from pathlib import Path
from generation.dexperts_generation import DExpertsGeneration

dexperts = DExpertsGeneration(
    base_model="gpt2",
    antiexpert_model="eliolio/gpt2-finetuned-redditbias",
    expert_model='gpt2',
    tokenizer="gpt2",
    seed=42,
)


def generate_dexperts(prompt, max_len=15, sample=True, filter_p=0.9, k=0, p=1.0, temperature=1.0, alpha=2.0):
    return dexperts.generate(prompt, max_len, sample, filter_p, k, p, temperature, alpha)


category = ['gender']
for cat in category:
    # Load gender prompt from json
    f = open(os.path.join('prompts/', cat + '_prompt.json'))

    # Load race prompt from json
    # f = open('./prompts/religious_ideology_prompt.json')
    # returns JSON object as a dictionary
    data = json.load(f)

# Iterating through the json list
    prompts = []
    subgroups = []
    names = []

    for subgroup in data:
        subgroups.append(subgroup)
        for name in data[subgroup]:
            names += name
            prompts += data[subgroup][name]

    text = {}
    for subgroup in data:
        text[subgroup] = []
        for person, prompts in tqdm(data[subgroup].items()[:10]):
            for prompt in prompts:
                prompt = prompt[:-1]  # remove space
                output = generate_dexperts(prompt)
                prohibitedWords = [person.replace('_', ' ')]
                big_regex = re.compile(
                    '|'.join(map(re.escape, prohibitedWords)))

                output = [big_regex.sub("XYZ", x) for x in output]
                text[subgroup] += output

    with open("results/test/bold_" + cat + "_dexperts_output.json", "w") as outfile:
        dictionary = text
        json.dump(dictionary, outfile)
