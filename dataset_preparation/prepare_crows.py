import pandas as pd
import json
import re

def get_diff_word(s1, s2):
    words1 = s1.lower().replace(".", "").split()
    words2 = s2.lower().replace(".", "").split()

    for w1, w2 in zip(words1, words2):
        if w1 != w2:
            return w1, w2
    return None, None

df = pd.read_csv("dataset_preparation/crows_pairs.csv")

gender_df = df[df['bias_type'] == 'gender'].copy()

populations = {"Stereotype_Terms": [], 
               "AntiStereotype_Terms": []}
examples = []

for idx, row in gender_df.iterrows():
    w1, w2 = get_diff_word(row['sent_more'], row['sent_less'])
    if w1 and w2:
        populations["Stereotype_Terms"].append(w1)
        populations["AntiStereotype_Terms"].append(w2)
        
        examples.append({"term": w1, 
                         "examples": [row['sent_more']],
                         "example_source": "CrowS-Pairs"})
        examples.append({"term": w2, 
                         "examples": [row['sent_less']], 
                         "example_source": "CrowS-Pairs"})

with open("populations_crows.json", "w") as f:
    json.dump(populations, f)

with open("crows_examples.txt", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Prepared {len(gender_df)} CrowS-Pairs for profiling.")