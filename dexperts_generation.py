from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
import re
from tqdm import tqdm
import json
import os
​
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModelForCausalLM.from_pretrained("gpt2")
# tokenizer = AutoTokenizer.from_pretrained("/home/gridsan/rlohanimit/fair-nlg/bold/biased_model")
# model = AutoModelForCausalLM.from_pretrained("/home/gridsan/rlohanimit/fair-nlg/biased_model")
tokenizer = AutoTokenizer.from_pretrained("/home/gridsan/rlohanimit/fair-nlg/bold/gender-biased_model")
model = AutoModelForCausalLM.from_pretrained("/home/gridsan/rlohanimit/fair-nlg/gender-biased_model")
​
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    out = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=15,
        do_sample=True,
        num_return_sequences=1,
        temperature=1.2,
        top_p=0.95,
        top_k = 40,
        pad_token_id=tokenizer.eos_token_id,
    )
    return [tokenizer.decode(out[i], skip_special_tokens=True) for i in range(len(out))]
​
​
category = ['gender']
for cat in category:
#Load gender prompt from json
    f = open( os.path.join('./prompts/',cat + '_prompt.json' ) )  
​
    #Load race prompt from json
    # f = open('./prompts/religious_ideology_prompt.json')  
    # returns JSON object as a dictionary
    data = json.load(f) 
​
# Iterating through the json list
    prompts = []
    subgroups = []
    names = []
​
    for subgroup in data:
        subgroups.append(subgroup)
        for name in data[subgroup]:
            names += name
            prompts += data[subgroup][name]
​
    text = {}
    for subgroup in data:
        text[subgroup] = []
        for person, prompts in tqdm(data[subgroup].items()):
            for prompt in prompts:
                prompt = prompt[:-1] # remove space
                output = generate_text(prompt)
                prohibitedWords = [person.replace('_',' ')]
                big_regex = re.compile('|'.join(map(re.escape, prohibitedWords)))
                
                output = [big_regex.sub("XYZ", x) for x in output]
                text[subgroup] += output
​
​
​
    with open("./results/test/bold_"+ cat +"_gpt2genderbias_output.json", "w") as outfile:
        dictionary = text
        json.dump(dictionary, outfile)