from nltk.corpus import wordnet as wn
import torch
from transformers import AutoTokenizer, AutoModel
import re
import numpy as np
from scipy.stats import ttest_ind
import nltk
from pathlib import Path
from typing import Dict, Optional

def get_word_idx(tokenizer, encoded_context, word) -> list[int]:
    """
    Robustly find token indices for `word` in `encoded_context` without using word_ids().
    Works for both fast and slow tokenizers.

    Strategy: decode tokens, strip '##', and find a contiguous subword span whose concatenation matches `word`.
    """
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
    """Retrieve contextual embedding for the given word in a context, for specified layers.
    Returns a tensor of shape [num_layers, hidden_dim]
    """
    encoded_context = tokenizer(context, return_tensors="pt", truncation=True)

    # Move tensors to model device (BatchEncoding supports .to())
    if hasattr(encoded_context, "to"):
        encoded_context = encoded_context.to(model.device)
    else:
        encoded_context = {k: v.to(model.device) for k, v in encoded_context.items()}

    with torch.no_grad():
        outputs = model(**encoded_context, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple: [embeddings_layer + transformer_layers]

    idxs = get_word_idx(tokenizer, encoded_context, word)
    if len(idxs) == 0:
        idxs = [0]  # fallback: CLS

    layerwise = []
    for layer in layers:
        vec = hidden_states[layer][0, idxs, :].mean(dim=0)
        layerwise.append(vec)

    return torch.stack(layerwise, dim=0)    
    
def load_model_for_embedding_retrieval(embedding_model_name, device, hf_token=None):

    # function to load model in eval mode and to output all hidden states

    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, use_fast = True, token = hf_token)
    
    try:
        embedding_model = AutoModel.from_pretrained(embedding_model_name, output_hidden_states=True, device_map="auto", token = hf_token)
    except:
        # for models where device map has not been implemented (and also not necessary, because these older models are small)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = AutoModel.from_pretrained(embedding_model_name, output_hidden_states=True, token = hf_token).to(device)
    
    embedding_model.eval();

    return tokenizer, embedding_model
    
def get_number_of_hidden_states(tokenizer, model):
    """Return number of hidden states including the embedding layer."""
    encoded = tokenizer("test", return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    with torch.no_grad():
        out = model(**encoded, output_hidden_states=True)
    return len(out.hidden_states)

def get_number_of_hidden_states_from_name(embedding_model_name, device):
    # function to get the number of hidden states from embedding_model_name

    tokenizer, embedding_model = load_model_for_embedding_retrieval(embedding_model_name, device)
    n_hidden_states = get_number_of_hidden_states(tokenizer, embedding_model)

    return n_hidden_states   



def get_examples_with_term(term, examples):
    """ Filter examples which contain the term of interested """
    
    return([example for example in examples if term in example.split(" ")])


### Wordnet Helper Functions


def get_wn_synset(term, pos, sense):
    """Get synset for a given term and a given sense.
    Return synset's name
    E.g., term="polite", sense=0 => synset="polite.a.01", name="polite"
    """
    wn_pos = None
    if pos == 'ADJECTIVE':
        wn_pos = wn.ADJ
    elif pos == 'NOUN':
        wn_pos = wn.NOUN

    wn_synset = wn.synsets(term, pos=wn_pos)[sense - 1]

    return wn_synset.name()
    
    
def get_wn_antonym(wn_synset):
    """Get the antonym of a given synset.
    """
    wn_synset = wn.synset(wn_synset)
    antonyms_lemmas = wn_synset.lemmas()[0].antonyms()

    antonym = np.NaN
    if len(antonyms_lemmas) > 0:
        antonym = antonyms_lemmas[0].synset().name()
    
    return antonym
    
    
def get_wn_definition(wn_synset):
    """Get WordNet's definition of a given synset.
    """
    wn_synset = wn.synset(wn_synset)
    return wn_synset.definition()


def get_wn_examples(wn_synset):
    """Get WordNet's sentence examples of a given synset in lower case.
    """
    wn_synset = wn.synset(wn_synset)
    lower_case_examples = list(map(str.lower, wn_synset.examples()))
    
    return lower_case_examples
    

def replace_wn_terms(term, example_list, synset):
    
    """
    Function to replace terms in example sentences.
    
    At times wordnet examples do not contain the term of interest but a synonym. 
    If the example contains the synset name, we replace it by the term. (e.g. “the right answer” for “Correct.a.01”)
    If term, synset name and the word in the example are three different tokens, we flag the term to change the example manually. (e.g. sentence = 'a very unsure young man', synset = diffident.a.02, term = timid)
    """

    are_3_diff_tokens = False
    altered_examples = []
    synset_word = synset.split('.')[0]

    for example in example_list:
        
        
        if term not in example and synset_word in example:
            # 1. Case where *term* NOT in *example* sentence, but its synset
            # e.g., 'stood apart with aloof dignity', synset = aloof.s.01, term = distant
            # => replace *synset word* in example sentence with given *term*
            new_sent = example.replace(synset_word, term)
            altered_examples.append(new_sent)
        
        else:
            # 2. Case for 3 different tokens: sentence = 'a very unsure young man', synset = diffident.a.02, term = timid
            # => add to list & manually change after
            if term not in example and synset_word not in example:
                are_3_diff_tokens = True

            # 3. Case for matches
            altered_examples.append(example)

    return altered_examples, are_3_diff_tokens
    

def get_all_noun_and_adjective_synsets(term):
    """ function to retrieve all synset names, definitions, examples for a term when we do not know the specific word sense.
    """

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


# Defining the templates for gendered terms or names

TEMPLATES = ['this is ', 'that is ', 'there is ', 'the person is ', 'here is ', ' is here', ' is there']

def fill_template(gendered_term, TEMPLATE, isNNP=False):
    """ Function to fill a template with a gendered term/name.
    """
    # if it is a noun
    if TEMPLATE.startswith(" is"):
        # add article "the" before noun
        if nltk.pos_tag([gendered_term])[0][1] in ['NN', 'JJ'] and isNNP is False:
            return('the ' + gendered_term + TEMPLATE)
        else:
            return(gendered_term + TEMPLATE)
    elif nltk.pos_tag([gendered_term])[0][1] in ['NN', 'JJ'] and isNNP is False:
        # add article "the" before noun
        TEMPLATE += 'the '
        return(TEMPLATE + gendered_term)
    else:
        return(TEMPLATE + gendered_term)
    

### simple function for statistical analysis

def get_stats(values):

    values = list(values)
    mean = np.mean(values)
    std = np.std(values)
    return mean, std

def standardize(values):

    values = list(values)
    mean, std = get_stats(values)
    return (values - mean)/std

def standardize_new_values(values, mean, std):

    values = list(values)
    return [(value- mean)/std for value in values]


def get_diff(values1,values2):
    
    values1 = list(values1)
    values2 = list(values2)
    
    mean1, std1 = get_stats(values1)
    mean2, std2 = get_stats(values2)
    diff = mean1 - mean2
    test_stats, diff_p = ttest_ind(values1, values2)
    diff_abs = abs(diff)
    
    return mean1, std1, mean2, std2, diff, diff_p, diff_abs

# --- Instruction conditioning helpers (project extension) ---
def load_instructions(path: str) -> Dict[str, str]:
    """Load instruction/persona definitions.

    Delegates to instructions.load_instructions, but kept here as a thin wrapper to
    avoid changing many imports elsewhere.
    """
    from instructions import load_instructions as _load
    return _load(Path(path))

def apply_instruction(context: str, instruction_text: Optional[str], template: str) -> str:
    """Apply an instruction prefix to a context string."""
    from instructions import apply_instruction_to_context as _apply
    return _apply(context=context, instruction_text=instruction_text, template=template)
