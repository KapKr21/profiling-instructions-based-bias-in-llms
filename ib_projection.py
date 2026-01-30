import numpy as np
import pandas as pd
import torch
import json as js
import nltk

nltk.download("averaged_perceptron_tagger_eng", 
              quiet=True)
import os
import warnings

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/Volumes/ExternalSSD/Dev/HuggingFace/hf_cache"

warnings.filterwarnings("ignore", 
                        category=FutureWarning, 
                        message=".*TRANSFORMERS_CACHE.*")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm
from scipy import linalg
import argparse
import utils
import matplotlib.pyplot as plt

import instructions

device = "auto" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("model", 
                    help="A model identifier string on https://huggingface.co", 
                    type=str)
parser.add_argument(
    "--populations",
    help="A json file containing a dictionary of population terms of format {population_name: list[str], population_name: list[str]}.",
    type=str,
    default="populations_names.json",
)
parser.add_argument(
    "--examples",
    help="A text file with context examples for terms in the stereotype dimensions dictionary or 'None' for no context.",
    type=str,
    default="generated_examples.txt",
)
parser.add_argument(
    "--hf_token",
    help="A user access token on https://huggingface.co to access gated models.",
    type=str,
    default=None,
)
parser.add_argument(
    "--standardization",
    help="Whether to standardize the results over the defined populations.",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument("--prefix", help="A prefix for the result files.", type=str, default="")

#iML (instructinos based) extension arguments
parser.add_argument(
    "--instructions",
    help="Path to instruction/persona definitions (JSON dict or TXT lines). If omitted, runs baseline only.",
    type=str,
    default=None,
)
parser.add_argument(
    "--instruction_template",
    help="Template used to prepend instruction to each context. Must contain {instruction} and {context}.",
    type=str,
    default=instructions.DEFAULT_INSTRUCTION_TEMPLATE,
)
parser.add_argument(
    "--instruction_scope",
    help="Where to apply instruction conditioning: "
    "'all' applies to stereotype dictionary contexts + population contexts; "
    "'population_only' applies only when projecting population terms (projection space fixed).",
    type=str,
    choices=["all", "population_only"],
    default="all",
)
parser.add_argument(
    "--embeddings_dir",
    help="Directory used to cache intermediate embeddings (created if missing).",
    type=str,
    default="./embeddings",
)
parser.add_argument(
    "--output_dir",
    help="Directory to write results (csv/pdf).",
    type=str,
    default=".",
)

args = parser.parse_args()

#Loading the dictionaries and specifying dimensions
stereodim_dictionary = pd.read_csv("./stereotype_dimensions_dictionary.csv", index_col=0)
stereotype_dimensions = ["Sociability", "Morality", "Ability", "Agency", "Status", "Politics", "Religion"]
warmth_competence_dimensions = {"Warmth": ["Sociability", "Morality"], "Competence": ["Ability", "Agency"]}
all_dimensions = stereotype_dimensions + list(warmth_competence_dimensions.keys())

#Loading the examples for embedding terms in the stereotype dimensions dictionary
if args.examples != "None":
    max_examples = 5
    context_examples = {}
    with open(args.examples, "r") as f:
        for idx, line in enumerate(f.readlines()):
            entry = js.loads(line)
            term = entry["term"]

            if "synset" in entry and entry["synset"]:
                term_key = f"{term} - {entry['synset']}"
            else:
                term_key = f"{term}_{idx}"
            
            context_examples[term_key] = entry["examples"][:max_examples]
    no_context = False
else:
    no_context = True

#Loading the population terms for projection
with open(args.populations, "r") as f:
    populations = js.load(f)
group1 = list(populations.keys())[0]
group2 = list(populations.keys())[1]

#Setting up the embedding model and tokenizer
tokenizer, embedding_model = utils.load_model_for_embedding_retrieval(args.model, device, hf_token=args.hf_token)
layers = [i for i in range(utils.get_number_of_hidden_states(tokenizer, embedding_model))]
base_model_name = args.model.split("/")[-1]

#Instruction setup (NEW)
if args.instructions is None:
    instruction_map = {"baseline": ""}
else:
    instruction_map = instructions.load_instructions(args.instructions)
    if "baseline" not in instruction_map:
        instruction_map = {"baseline": ""} | instruction_map

def _apply_instruction(context: str, 
                       instruction_text: str) -> str:
    return instructions.apply_instruction_to_context(
        context=context,
        instruction_text=instruction_text,
        template=args.instruction_template,
    )

def run_for_instruction(instruction_name: str, 
                        instruction_text: str) -> None:
    safe_instruction = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in instruction_name.strip()])
    pop_name = os.path.splitext(os.path.basename(args.populations))[0]

    run_name = f"{safe_instruction}_{pop_name}"

    model_output_dir = os.path.join("results", base_model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    population_only = args.instruction_scope == "population_only"
    build_dictionary = (not population_only) or (instruction_name == "baseline")
    axes_run_name = f"{base_model_name}__baseline" if population_only else run_name

    os.makedirs(args.embeddings_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    for layer in layers:
        os.makedirs(os.path.join(args.embeddings_dir, 
                                 f"{axes_run_name}-L{layer}"), 
                                 exist_ok=True)

    for layer in layers:
        os.makedirs(os.path.join(args.embeddings_dir, 
                                 f"{run_name}-L{layer}"), 
                                 exist_ok=True)

    #Retrieving sense embeddings and embeddings of stereotype dimensions
    if build_dictionary:
        embedding_dict = {
            layer: {"sense_embeddings": [], 
                    "sense_embedding_labels": [], 
                    "pole_embedding_dict": {}}
            for layer in layers
        }

        apply_instr_to_dict = (args.instruction_scope == "all")

        for _, row in tqdm(
            stereodim_dictionary.iterrows(),
            total=len(stereodim_dictionary),
            desc=f"[{instruction_name}] Retrieving embeddings for stereotype dimensions.",
        ):
            term = row["term"]
            synset = row["synset"]
            dimension = row["dimension"]
            direction = row["dir"]

            key = f"{term} - {synset}"
            contexts = context_examples.get(key, context_examples.get(term, []))

            if len(contexts) == 0:
                contexts = [term]

            if apply_instr_to_dict and (not no_context):
                contexts = [_apply_instruction(c, instruction_text) for c in contexts]

            layerwise_sense_embeddings = [
                utils.get_word_embedding_by_layer(tokenizer, embedding_model, context, term, layers)
                for context in contexts
            ]
            layerwise_sense_embeddings = torch.stack(layerwise_sense_embeddings).mean(dim=0)

            for layer, layer_dict in embedding_dict.items():
                sense_embedding = layerwise_sense_embeddings[layer]
                key = dimension + "-" + direction
                layer_dict["pole_embedding_dict"].setdefault(key, []).append(sense_embedding)
                layer_dict["sense_embeddings"].append(sense_embedding)
                layer_dict["sense_embedding_labels"].append(f"{term} - {synset} ({dimension}-{direction})")

        #Saving pole embeddings into the AXIS RUN directory (baseline if population_only)
        for layer, layer_dict in embedding_dict.items():
            layer_dir = os.path.join(args.embeddings_dir, f"{axes_run_name}-L{layer}")

            for key, vecs in layer_dict["pole_embedding_dict"].items():
                out_path = os.path.join(layer_dir, f"{key}_embeddings.npy")
                with open(out_path, "wb") as f:
                    np.save(f, torch.stack(vecs).cpu().numpy())

            with open(os.path.join(layer_dir, "sense_embedding_labels.txt"), "w") as f:
                for lab in layer_dict["sense_embedding_labels"]:
                    f.write(lab + "\n")
    else:
        layer0_dir = os.path.join(args.embeddings_dir, f"{axes_run_name}-L0")
        if not os.path.isdir(layer0_dir):
            raise FileNotFoundError(
                f"population_only requires baseline axes to exist at {layer0_dir}. "
                f"Run baseline first (it should be included in instructions.json) to create them."
            )

    #Base Change Matrices for Projection
    warmth_competence_base_change_inv = {}
    stereodim_base_change_inv = {}

    for layer in tqdm(layers, desc=f"[{instruction_name}] Preparing layerwise base change matrices."):
        layer_dir = os.path.join(args.embeddings_dir, f"{axes_run_name}-L{layer}")

        stereodim_base_change = []
        for dim in stereotype_dimensions:
            with open(os.path.join(layer_dir, f"{dim}-low_embeddings.npy"), "rb") as f:
                low_pole_embeddings = np.load(f)
            with open(os.path.join(layer_dir, f"{dim}-high_embeddings.npy"), "rb") as f:
                high_pole_embeddings = np.load(f)

            dim_low_mean_value = np.average(low_pole_embeddings, axis=0)
            dim_high_mean_value = np.average(high_pole_embeddings, axis=0)
            stereodim_base_change.append(dim_high_mean_value - dim_low_mean_value)

        with open(os.path.join(layer_dir, "stereodim_base_change.npy"), "wb") as f:
            np.save(f, stereodim_base_change)
        stereodim_base_change_inv[layer] = linalg.pinv(np.transpose(np.vstack(stereodim_base_change)))

        warmth_competence_base_change = []
        for wc_dim, subdims in warmth_competence_dimensions.items():
            low_vectors = []
            high_vectors = []
            for subdim in subdims:
                with open(os.path.join(layer_dir, f"{subdim}-low_embeddings.npy"), "rb") as f:
                    low_vectors.append(np.load(f))
                with open(os.path.join(layer_dir, f"{subdim}-high_embeddings.npy"), "rb") as f:
                    high_vectors.append(np.load(f))

            low_mean = np.mean(np.vstack(low_vectors), axis=0)
            high_mean = np.mean(np.vstack(high_vectors), axis=0)
            warmth_competence_base_change.append(high_mean - low_mean)

        with open(os.path.join(layer_dir, "warmth_competence_base_change.npy"), "wb") as f:
            np.save(f, warmth_competence_base_change)
        warmth_competence_base_change_inv[layer] = linalg.pinv(np.transpose(np.vstack(warmth_competence_base_change)))

    #Projectioning of populations to stereotype dimensions
    result_dict = {}
    for layer in layers:
        result_dict[f"{run_name}-L{layer}"] = {
            "Model": len(list(warmth_competence_dimensions.keys()) + stereotype_dimensions) * [f"{run_name}-L{layer}"],
            "Dimension": list(warmth_competence_dimensions.keys()) + stereotype_dimensions,
        }

    apply_instr_to_pop = (instruction_text.strip() != "")

    for group, terms in populations.items():
        for term in tqdm(terms, desc=f"[{instruction_name}] Projecting {group} to stereotype dimensions"):
            isNNP = True if nltk.pos_tag([term])[0][1] == "NNP" or "names" in group.lower() else False

            contexts = [term] if no_context else [
                utils.fill_template(term, TEMPLATE, isNNP=isNNP) for TEMPLATE in utils.TEMPLATES
            ]
            if (not no_context) and apply_instr_to_pop:
                contexts = [_apply_instruction(c, instruction_text) for c in contexts]

            layerwise_sense_embeddings = [
                utils.get_word_embedding_by_layer(tokenizer, embedding_model, context, term, layers)
                for context in contexts
            ]
            layerwise_sense_embeddings = torch.stack(layerwise_sense_embeddings).mean(dim=0)

            for layer in layers:
                sense_embedding = layerwise_sense_embeddings[layer]
                warmth_competence_embedding = torch.matmul(
                    torch.from_numpy(warmth_competence_base_change_inv[layer]).double(),
                    sense_embedding.double(),
                ).numpy()
                stereodim_embedding = torch.matmul(
                    torch.from_numpy(stereodim_base_change_inv[layer]).double(),
                    sense_embedding.double(),
                ).numpy()

                result_dict[f"{run_name}-L{layer}"][term] = np.concatenate((warmth_competence_embedding, stereodim_embedding))

    results = pd.concat([pd.DataFrame.from_dict(result_dict[k]) for k in result_dict.keys()])

    #Averaging over layers for each dimension
    for dimension in all_dimensions:
        new_row = [run_name] + [dimension] + list(
            results.loc[results["Dimension"] == dimension].set_index(["Model", "Dimension"]).mean(axis=0)
        )
        new_row = pd.DataFrame([new_row], columns=results.columns)
        results = pd.concat((results, new_row))

    if len(populations.keys()) > 2:
        print(f"More than two populations in {args.populations}, the analysis is run to compare the first two groups.")

    if args.standardization is True:
        results[populations[group1] + populations[group2]] = results.apply(
            lambda x: utils.standardize(x[populations[group1] + populations[group2]]),
            axis="columns",
            result_type="expand",
        )

    stats = pd.DataFrame(
        results.apply(
            lambda x: utils.get_diff(x[populations[group1]], x[populations[group2]]),
            axis="columns",
            result_type="expand",
        )
    )
    stats.columns = [
        f"{group1}_mean",
        f"{group1}_std",
        f"{group2}_mean",
        f"{group2}_std",
        "diff",
        "diff_pvalue",
        "diff_abs",
    ]
    results = pd.concat([results, stats], axis=1)

    csv_filename = f"{args.prefix}{run_name}_projection_results.csv"
    csv_path = os.path.join(model_output_dir, csv_filename)
    results.to_csv(csv_path, index=False)
    print(f"[{instruction_name}] Results for {pop_name} saved in: {csv_path}")

    #Plotting warmth and competence
    plot_dimensions = ["Competence", "Warmth"]
    polar_labels = {
        "Warmth": {"low": "Low Warmth", "high": "High Warmth"},
        "Competence": {"low": "Low Competence", "high": "High Competence"},
    }

    fig, ax1 = plt.subplots(1, 1)

    styles = {
        "line": {group1: "solid", group2: "dashdot"},
        "color": {group1: "rebeccapurple", group2: "mediumorchid"},
    }

    plot_values = {group1: [], group2: []}
    left_labels = []
    right_labels = []
    bold_indices = []

    for i, dim in enumerate(plot_dimensions):
        row = results.loc[(results["Model"] == run_name) & (results["Dimension"] == dim)]
        if row.empty: continue
        
        for g in [group1, group2]:
            plot_values[g].append(float(row[f"{g}_mean"].values[0]))

        low_lab = polar_labels[dim]["low"]
        high_lab = polar_labels[dim]["high"]

        if float(row["diff_pvalue"].values[0]) < 0.05:
            high_lab += "$^*$"
            bold_indices.append(i)
        
        left_labels.append(low_lab)
        right_labels.append(high_lab)

    y_positions = np.arange(len(plot_dimensions))
    ax1.axvline(0, linewidth=0.6, color="gray", alpha=0.6)

    for g in [group1, group2]:
        ax1.plot(
            plot_values[g],
            y_positions,
            linestyle=styles["line"][g],
            color=styles["color"][g],
            marker="o",
            label=g,
        )

    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(left_labels, fontsize=9)
    for i, label in enumerate(ax1.get_yticklabels()):
        if i in bold_indices:
            label.set_fontweight("bold")

    ax1.set_xlim(-1.0, 1.0)
    ax1.set_xlabel("projected values", fontsize=8)

    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(right_labels, fontsize=9)
    for i, label in enumerate(ax2.get_yticklabels()):
        if i in bold_indices:
            label.set_fontweight("bold")

    ax1.set_title(f"{base_model_name} ({instruction_name})", fontsize=9)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=2, fontsize=9)

    fig.set_figwidth(3.5)
    fig.set_figheight(1.3)

    pdf_filename = f"{args.prefix}{run_name}_warmth_competence_profile.pdf"
    pdf_path = os.path.join(model_output_dir, pdf_filename)
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"[{instruction_name}] Profile for {pop_name} saved in: {pdf_path}")
    plt.close(fig)

for instr_name, instr_text in instruction_map.items():
    run_for_instruction(instr_name, instr_text)