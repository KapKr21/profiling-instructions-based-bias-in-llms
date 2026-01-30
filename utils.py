from nltk.corpus import wordnet as wn
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, T5EncoderModel
import re
import numpy as np
from scipy.stats import ttest_ind
import nltk
from pathlib import Path
from typing import Dict, Optional

def get_word_idx(tokenizer, encoded_context, word) -> list[int]:
    decoded = tokenizer.convert_ids_to_tokens(encoded_context["input_ids"][0])
    decoded_cleaned = [t.replace("##", "") for t in decoded]

    word_clean = re.sub(r"\s+", "", word.strip())

    for i in range(len(decoded_cleaned)):
        concat = ""
        span = []
        for j in range(i, min(i + 12, len(decoded_cleaned))):
            concat += decoded_cleaned[j]
            span.append(j)
            if concat.lower() == word_clean.lower():
                return span
    return []
    
def get_word_embedding_by_layer(tokenizer, model, context, word, layers):
    encoded_context = tokenizer(context, return_tensors="pt", truncation=True)

    if hasattr(encoded_context, "to"):
        encoded_context = encoded_context.to(model.device)

    with torch.no_grad():
        if "t5" in model.config.model_type.lower():
            if hasattr(model, "encoder"):
                outputs = model.encoder(**encoded_context, output_hidden_states=True)
            else:
                outputs = model(**encoded_context, output_hidden_states=True)
        else:
            outputs = model(**encoded_context, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    idxs = get_word_idx(tokenizer, encoded_context, word)
    if len(idxs) == 0:
        idxs = [0]

    layerwise = []
    for layer in layers:
        vec = hidden_states[layer][0, idxs, :].mean(dim=0)
        layerwise.append(vec)

    return torch.stack(layerwise, dim=0)    
    
def load_model_for_embedding_retrieval(embedding_model_name, device, hf_token=None):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, 
                                              use_fast=True, 
                                              token=hf_token)
    config = AutoConfig.from_pretrained(embedding_model_name, 
                                        token=hf_token)

    if getattr(config, "model_type", None) == "t5":
        model = T5EncoderModel.from_pretrained(embedding_model_name, 
                                               token=hf_token)
    else:
        model = AutoModel.from_pretrained(embedding_model_name, 
                                          token=hf_token)

    model.to(device)
    model.eval()
    return tokenizer, model
    
def get_number_of_hidden_states(tokenizer, model):
    encoded = tokenizer("test", return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    with torch.no_grad():
        out = model(**encoded, output_hidden_states=True)
    return len(out.hidden_states)

def get_number_of_hidden_states_from_name(embedding_model_name, device):
    tokenizer, embedding_model = load_model_for_embedding_retrieval(embedding_model_name, device)
    n_hidden_states = get_number_of_hidden_states(tokenizer, embedding_model)

    return n_hidden_states   

def get_examples_with_term(term, examples):
    return([example for example in examples if term in example.split(" ")])

def get_wn_synset(term, pos, sense):
    wn_pos = None
    if pos == 'ADJECTIVE':
        wn_pos = wn.ADJ
    elif pos == 'NOUN':
        wn_pos = wn.NOUN

    wn_synset = wn.synsets(term, pos=wn_pos)[sense - 1]

    return wn_synset.name()
    
def get_wn_antonym(wn_synset):
    wn_synset = wn.synset(wn_synset)
    antonyms_lemmas = wn_synset.lemmas()[0].antonyms()

    antonym = np.NaN
    if len(antonyms_lemmas) > 0:
        antonym = antonyms_lemmas[0].synset().name()
    
    return antonym
    
def get_wn_definition(wn_synset):
    wn_synset = wn.synset(wn_synset)
    return wn_synset.definition()

def get_wn_examples(wn_synset):
    wn_synset = wn.synset(wn_synset)
    lower_case_examples = list(map(str.lower, wn_synset.examples()))
    
    return lower_case_examples
    
def replace_wn_terms(term, example_list, synset):
    are_3_diff_tokens = False
    altered_examples = []
    synset_word = synset.split('.')[0]

    for example in example_list:
        if term not in example and synset_word in example:
            new_sent = example.replace(synset_word, term)
            altered_examples.append(new_sent)
        else:
            if term not in example and synset_word not in example:
                are_3_diff_tokens = True

            altered_examples.append(example)

    return altered_examples, are_3_diff_tokens
    
def get_all_noun_and_adjective_synsets(term):
    synsets = [wn_synset.name() for wn_synset in wn.synsets(term) if wn_synset.pos() in ["n", "a", "s"]]
    definitions = []
    examples = []

    for synset in synsets:
        synset_examples = list(map(str.lower, wn.synset(synset).examples()))
        synset_examples, diff = replace_wn_terms(term, synset_examples, synset)
        synset_examples = [example for example in synset_examples if term in example]
        examples = examples + synset_examples
    
        definitions.append(wn.synset(synset).definition())
    
    return(synsets, definitions, examples)

TEMPLATES = ['this is ', 
             'that is ', 
             'there is ', 
             'the person is ', 
             'here is ', 
             ' is here', 
             ' is there']

def fill_template(gendered_term, TEMPLATE, isNNP=False):
    if TEMPLATE.startswith(" is"):
        if nltk.pos_tag([gendered_term])[0][1] in ['NN', 'JJ'] and isNNP is False:
            return('the ' + gendered_term + TEMPLATE)
        else:
            return(gendered_term + TEMPLATE)
    elif nltk.pos_tag([gendered_term])[0][1] in ['NN', 'JJ'] and isNNP is False:
        TEMPLATE += 'the '
        return(TEMPLATE + gendered_term)
    else:
        return(TEMPLATE + gendered_term)
    
def get_stats(values):
    values = list(values)
    mean = np.mean(values)
    std = np.std(values)
    return mean, std

def standardize(values):
    values = np.array(list(values), dtype=float)
    mean = np.mean(values)
    std = np.std(values)

    if std == 0 or np.isnan(std):
        return np.zeros_like(values)

    return (values - mean) / std

def standardize_new_values(values, mean, std):
    values = list(values)
    return [(value- mean)/std for value in values]

def get_diff(values1, values2):
    values1 = np.array(list(values1), dtype=float)
    values2 = np.array(list(values2), dtype=float)
    
    mean1, std1 = get_stats(values1)
    mean2, std2 = get_stats(values2)
    diff = mean1 - mean2
    
    if np.allclose(values1, values1[0]) and np.allclose(values2, values2[0]):
        test_stats, diff_p = 0.0, 1.0 if np.isclose(mean1, mean2) else 0.0
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            test_stats, diff_p = ttest_ind(values1, values2)
            
    diff_abs = abs(diff)
    
    return mean1, std1, mean2, std2, diff, diff_p, diff_abs

def load_instructions(path: str) -> Dict[str, str]:
    from instructions import load_instructions as _load
    return _load(Path(path))

def apply_instruction(context: str, 
                      instruction_text: Optional[str], template: str) -> str:
    from instructions import apply_instruction_to_context as _apply
    return _apply(context=context, 
                  instruction_text=instruction_text, 
                  template=template)