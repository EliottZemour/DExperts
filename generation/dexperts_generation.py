from pathlib import Path
from typing import Union, List

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, modeling_utils, GPT2PreTrainedModel
from generation.gpt2_generation import GPT2Generation

from utils import utils
from utils.generation_utils import top_k_top_p_filtering

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

class DExpertsGeneration(GPT2Generation): 
    STOP_TOKEN = "<|endoftext|>"

    def __init__(
        self, 
        base_model: Union[str, Path, GPT2PreTrainedModel],
        antiexpert_model: Union[str, Path, GPT2PreTrainedModel] = None,
        expert_model: Union[str, Path, GPT2PreTrainedModel] = None,
        tokenizer: str = 'gpt2', 
        seed: int = 42,
    ):
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        utils.set_seed(seed, n_gpu)

        self.base_model = GPT2LMHeadModel.from_pretrained(base_model).to(self.device)
        
        if antiexpert_model:
            self.antiexpert = GPT2LMHeadModel.from_pretrained(antiexpert_model, use_auth_token=True).to(self.device)
        else:
            self.antiexpert = None
        
        if expert_model:
            self.expert = GPT2LMHeadModel.from_pretrained(expert_model, use_auth_token=True).to(self.device)
        else:
            self.expert = None
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer, pad_token=self.STOP_TOKEN)
        assert self.tokenizer.eos_token_id == self.tokenizer.pad_token_id

    def __repr__(self):
        return f'<DExpertsGenerator model_name_or_path="{self.model}">'

    def generate(self,
                 prompt: Union[str, List[str]],
                 max_len: int = 20,
                 sample: bool = True,
                 filter_p: float = 0.9,
                 k: int = 0,
                 p: float = 1.0,
                 temperature: float = 1.0,
                 alpha: float = 0.0,
                 **model_kwargs):
        if isinstance(prompt, str):
            prompt = [prompt]

        encodings_dict = self.tokenizer.batch_encode_plus(prompt, pad_to_max_length=True, return_tensors='pt')

        input_ids = encodings_dict['input_ids'].to(self.device)
        attention_mask = encodings_dict['attention_mask'].to(self.device)
        batch_size, input_seq_len = input_ids.shape

        position_ids = attention_mask.cumsum(dim=1) - 1
        unfinished_sents = torch.ones(batch_size, dtype=torch.long, device=self.device)

        self.base_model.eval()
        if self.expert:
            self.expert.eval()
        if self.antiexpert:
            self.antiexpert.eval()
        with torch.no_grad():
            # max_len = 1
            for step in range(max_len):
                # base model prediction
                base_logits = self.base_model(
                    input_ids, attention_mask=attention_mask, position_ids=position_ids, **model_kwargs).logits
                
                # expert prediction
                if self.expert:
                    expert_logits = self.expert(
                        input_ids, attention_mask=attention_mask, position_ids=position_ids, **model_kwargs).logits
                else:
                    expert_logits = base_logits
                
                # antiexpert prediction
                if self.antiexpert:
                    antiexpert_logits = self.antiexpert(
                        input_ids, attention_mask=attention_mask, position_ids=position_ids, **model_kwargs).logits
                else:
                    antiexpert_logits = base_logits
        
                #return base_logits, expert_logits, antiexpert_logits
                # print(type(base_logits))
                # print(base_past.isna)
                # print(base_logits.shape, expert_logits.shape, antiexpert_logits.shape)
                
                if filter_p < 1.0:
                    base_logits = top_k_top_p_filtering(base_logits, top_p=filter_p)
                # print(f"{expert_logits.isnan().any()=}")
                # print(f"{antiexpert_logits.isnan().any()=}")
                # DExperts
                # alpha = torch.tensor(alpha).to(self.device)
                # tmp = base_logits - base_logits
                # print(alpha)
                # print(tmp)
                # print(tmp.isnan().any())
                if self.antiexpert is not None or self.expert is not None:
                    ensemble_logits = base_logits + alpha * (expert_logits - antiexpert_logits)
                else:
                    ensemble_logits = base_logits #+ alpha * (expert_logits - antiexpert_logits)

                # in the first decoding step, we want to use the 'real' last position for each sentence
                if step == 0:
                    last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
                    next_token_logits = ensemble_logits[range(batch_size), last_non_masked_idx, :]
                else:
                    next_token_logits = ensemble_logits[:, -1, :]

                if sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    if k > 0 or p < 1.0:
                        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=k, top_p=p)
                    # Sample
                    probs = F.softmax(next_token_logits, dim=-1)
                    
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # either append a padding token here if <EOS> has been seen or append next token
                tokens_to_add = next_tokens * unfinished_sents + self.tokenizer.pad_token_id * (1 - unfinished_sents)

                # this updates which sentences have not seen an EOS token so far
                # if one EOS token was seen the sentence is finished
                eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id
                unfinished_sents.mul_((~eos_in_sents).long())

                # stop when there is an EOS in each sentence
                if unfinished_sents.max() == 0:
                    break

                # Update input_ids, attention_mask and position_ids
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)
                position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                           for output in input_ids[:, :]]
        return decoded_outputs

    def get_logits(self, encodings_dict, alpha=2.0):
        self.base_model.eval()
        if self.expert:
            self.expert.eval()
        if self.antiexpert:
            self.antiexpert.eval()
        
        with torch.no_grad():

            # base model prediction
            base_logits = self.base_model(encodings_dict).logits
            
            # expert prediction
            if self.expert:
                expert_logits = self.expert(encodings_dict).logits
            else:
                expert_logits = base_logits
            
            # antiexpert prediction
            if self.antiexpert:
                antiexpert_logits = self.antiexpert(encodings_dict).logits
            else:
                antiexpert_logits = base_logits
    

            if self.antiexpert is not None or self.expert is not None:
                ensemble_logits = base_logits + alpha * (expert_logits - antiexpert_logits)
            else:
                ensemble_logits = base_logits #+ alpha * (expert_logits - antiexpert_logits)

        return ensemble_logits