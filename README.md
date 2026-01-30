
# Instruction-Conditioned Bias Profiling in Language Models

This repository provides a framework for profiling social bias in contextual word embeddings using stereotype dimensions derived from social psychology. The pipeline supports instruction-conditioned analysis, allowing users to study how personas (e.g., conservative, liberal, leftist) reshape internal bias representations across different models (Llama-3.2-1B-Instruct, BERT-base-uncased, and Flan-T5).

The code is adapted from prior work on stereotype-based embedding analysis and extended for interpretability-focused machine learning (iML) experiments.

### Overview

Language models encode social associations that can surface as biased behavior. Instead of evaluating bias only through downstream tasks, this project examines representation-level bias by:

- Defining interpretable stereotype dimensions (e.g. sociability, morality, ability, agency, status, politics, and religion)

- Embedding population terms (names, gendered terms, CrowS-Pairs dataset targets)

- Projecting those embeddings onto stereotype axes

- Visualising results as:

  -- Warmth–Competence summaries

The instruction-conditioned extension enables analysis of how personas change these internal representations.

### Citation

This repository builds on the methodology introduced in the following paper.

~~~
@inproceedings{
schuster2024profiling,
title={Profiling Bias in {LLM}s: Stereotype Dimensions in Contextual Word Embeddings},
author={Carolin M. Schuster and Maria-Alexandra Dinisor and Shashwat Ghatiwala and Georg Groh},
booktitle={The Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies},
year={2024}
}
~~~

# Usage

Environment
```
conda env create -f ib-profiling-env.yaml
conda activate ib-profiling-env
```

### Instructions based Profiling (iML Extension)

This fork extends the original pipeline by prepending instructions or personas to every context passed to the model.

Following instruction scope is supported:

- all: instructions affect both stereotype axes and populations

### Command Categories

Below, commands are grouped by dataset type and model (Llama-3.2-1B-Instruct, BERT-base-uncased, and Flan-T5).

- CrowS-Pairs–based Populations

Used to study implicit stereotypes underlying sentence-pair bias benchmarks.

FLAN-T5

```
python ib_projection.py google/flan-t5-base \
  --populations populations_crows.json \
  --examples crows_examples.txt \
  --instructions instructions.json \
  --instruction_scope all
```

LLaMA-3.2-1B-Instruct

```
python ib_projection.py meta-llama/Llama-3.2-1B-Instruct \
  --populations populations_crows.json \
  --examples crows_examples.txt \
  --instructions instructions.json \
  --instruction_scope all \
  --hf_token YOUR_HF_TOKEN
```

BERT

```
python ib_projection.py google-bert/bert-base-uncased \
  --populations populations_crows.json \
  --examples crows_examples.txt \
  --instructions instructions.json \
  --instruction_scope all
```

- Gendered Terms (WEAT-style)

Used for controlled lexical bias analysis.

```
python ib_projection.py google/flan-t5-base \
  --populations populations_terms.json \
  --examples generated_examples.txt \
  --instructions instructions.json \
  --instruction_scope all
```

```
python ib_projection.py meta-llama/Llama-3.2-1B-Instruct \
  --populations populations_terms.json \
  --examples generated_examples.txt \
  --instructions instructions.json \
  --instruction_scope all \
  --hf_token YOUR_HF_TOKEN
```

```
python ib_projection.py google-bert/bert-base-uncased \
  --populations populations_terms.json \
  --examples generated_examples.txt \
  --instructions instructions.json \
  --instruction_scope all
```

- Name-Based Populations

Used to probe demographic associations through personal names.

```
python ib_projection.py google/flan-t5-base \
  --populations populations_names.json \
  --examples generated_examples.txt \
  --instructions instructions.json \
  --instruction_scope all
```

```
python ib_projection.py meta-llama/Llama-3.2-1B-Instruct \
  --populations populations_names.json \
  --examples generated_examples.txt \
  --instructions instructions.json \
  --instruction_scope all \
  --hf_token YOUR_HF_TOKEN
```

```
python ib_projection.py google-bert/bert-base-uncased \
  --populations populations_names.json \
  --examples generated_examples.txt \
  --instructions instructions.json \
  --instruction_scope all
```

### Outputs

Each run produces:

*_projection_results.csv
→ layer-wise and dimension-wise projection scores

*_warmth_competence_profile.pdf
→ 2D summary plot (Warmth vs Competence)

### Instruction File Format

Instructions are defined as a JSON dictionary:

{
  "baseline": "",
  "conservative": "You are conservative.",
  "liberal": "You are liberal.",
  "leftist": "You are leftist."
}

Each instruction is treated as a semantic intervention on the model’s embedding space.

### Notes on Instruction Scope

--instruction_scope all
Recomputes stereotype axes under each instruction
Recommended for interpretability analysis

--instruction_scope population_only
Keeps stereotype axes fixed to baseline
Useful for controlled comparisons

### Resources

- Stereotype Dictionaries

Based on the Stereotype Content Model:

Seed dictionary: https://osf.io/ghfkb

Full dictionary: https://osf.io/m9nb5

- Vocabulary Sources

Names
US Social Security Administration, top names over the last 100 years.

Gendered Terms
Adapted from WEAT-style experiments (Caliskan et al., 2017).

### References

Caliskan et al. (2017). Semantics derived automatically from language corpora contain human-like biases. Science.

Engler et al. (2022). SensePOLAR. EMNLP Findings.

Fraser et al. (2021). Understanding and Countering Stereotypes. ACL-IJCNLP.

Mathew et al. (2020). The Polar Framework. WWW.

Nicolas et al. (2021). Comprehensive stereotype content dictionaries. EJSP.