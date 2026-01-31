
# Profiling Instructions-Based Bias in Large Language Models (iML Project 2025/26)

This repository contains a framework for profiling social bias in contextual word embeddings using stereotype dimensions derived from social psychology. Building upon original research in stereotype-based embedding analysis, this fork introduces an instruction-conditioned analysis pipeline.

This extension allows researchers to study how specific personas (e.g., conservative, liberal, or leftist) shift the internal bias representations within state-of-the-art models like Llama-3.2, BERT, and Flan-T5.

### Overview

Language Models (LMs) encode complex social associations that often manifest as biased behavior. This project moves beyond downstream task evaluation to examine representation-level bias by:

- Defining Interpretable Dimensions: Utilizing social psychology axes such as sociability, morality, ability, agency, status, politics, and religion.

- Embedding Populations: Mapping demographic terms (names, gendered terms, and CrowS-Pairs targets) into the model's latent space.

- Geometric Projection: Projecting population embeddings onto predefined stereotype axes.

- Visualizing Shifts: Generating Warmth–Competence summaries to observe how different personas alter the model's "worldview."

### Citation

This repository is an extension of the methodology introduced in the following paper. If you use this code, please cite:

```
@inproceedings{
  schuster2024profiling,
  title={Profiling Bias in {LLM}s: Stereotype Dimensions in Contextual Word Embeddings},
  author={Carolin M. Schuster and Maria-Alexandra Dinisor and Shashwat Ghatiwala and Georg Groh},
  booktitle={The Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies},
  year={2024}
}
```

## Getting Started

- Install the open-source-distribution [anaconda](https://www.anaconda.com/download).

- Create a new environment with python 3.9 using following commands and activate it.

```
conda env create -f ib-profiling-env.yaml
conda activate ib-profiling-env
```

### Model Weights

We use pretrained checkpoints from the Hugging Face Transformers library. All models are used in inference-only mode and no fine-tuning is performed.

- Decoder only: meta-llama/Llama-3.2-1B-Instruct (Requires HF access token)

- Encoder-only: google-bert/bert-base-uncased

- Encoder-Decoder (Seq2Seq): google/flan-t5-base


## Instructions-Based Profiling (iML Extension)

This extension prepends specific instructions or personas to every context passed to the model.

### Instruction Scope

   - --instruction_scope all: The provided instructions influence both the stereotype axes and the population embeddings, allowing for a complete re-centering of the model's semantic space based on the persona.

### Instruction Configuration

Instructions are defined in a instructions.json file. Example format:

```
{
  "baseline": "",
  "conservative": "You are conservative.",
  "liberal": "You are liberal.",
  "leftist": "You are leftist."
}
```

### Execution Guide

The following commands demonstrate how to run the profiling pipeline across different datasets and models.

1. CrowS-Pairs Populations

Analyzing implicit stereotypes within sentence-pair bias benchmarks.

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

2. Gendered Terms (WEAT-style)

Using for controlled lexical bias analysis using gender-specific terminology.

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

3. Name-Based Populations

Probes demographic associations using common names from the US Social Security Administration.

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

Each execution generates two primary artifacts:

1. *_projection_results.csv: Contains layer-wise and dimension-wise projection scores for detailed statistical analysis.

2. *_warmth_competence_profile.pdf: A 2D scatter plot visualizing the population's position on the Warmth vs. Competence axes.

## Resources & References

### Data Sources

- CrowS-Pairs Dataset: A challenge [dataset](nyu-mll/crows-pairs) for measuring social biases in language models.

- Stereotype Dictionaries: Based on the Stereotype Content Model [Seed](https://osf.io/ghfkb) | [Full](https://osf.io/m9nb5).

- Vocabulary: Names sourced from the US Social Security Administration and Gendered terms adapted from Caliskan et al. (2017).

### Key References

- Schuster et al. (2024) Profiling bias in llms.

- Engler et al. (2022): SensePOLAR framework.

- Nicolas et al. (2021): Comprehensive stereotype content dictionaries.

- Fraser et al. (2021). Understanding and Countering Stereotypes. ACL-IJCNLP.

- Mathew et al. (2020). The Polar Framework.

- Caliskan et al. (2017): Semantics derived automatically from language corpora contain human-like biases.