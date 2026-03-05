import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL_NAME = "gpt-oss:20b"

INPUT_FILE  = "paper_other.jsonl"
OUTPUT_FILE = "paper_other_classified.jsonl"
MAX_WORKERS = 3
AUTOSAVE_INTERVAL = 100

SUGGESTED_CATEGORIES = [
    "pipeline_design", "qc_preprocessing", "sequencing", "host_decontamination",
    "alignment", "assembly", "assembly_qc", "binning", "bin_qc", "taxonomy",
    "annotation", "functional_profiling", "quantification", "diversity_analysis",
    "statistical_analysis", "visualization", "machine_learning", "multiomics",
    "genomics_infra", "experimental_design", "sample_collection", "phylogenetics",
    "dna_extraction", "troubleshooting", "preprocessing", "general_question",
    "literature_review", "validation", "benchmark", "amplicon_sequencing",
    "genome_editing", "epigenetic_analysis", "metabolomics", "metabolite_analysis",
    "metabolic_pathway", "biogeochemistry", "biochemistry", "enzyme_activity",
    "microbial_physiology", "microbial_interaction", "microbial_isolation",
    "biodegradation", "bioremediation", "cell_biology", "cell_culture",
    "strain_isolation", "antibiotic_resistance_analysis", "antimicrobial_mechanism",
    "mechanisms_of_resistance", "detection", "environmental_analysis",
    "environmental_microbiology", "ecology", "sampling_design", "soil_analysis",
    "geochemistry", "nutrient_cycling", "wastewater_treatment", "pollution_assessment",
    "phage_therapy", "phage_biology", "virus_host_interaction", "pathogen_transmission",
    "classification", "morphology", "interaction_analysis", "dysbiosis_association",
    "tissue_processing", "extraction", "purification", "plant_biology",
    "habitat_assessment", "epidemiology", "surveillance", "risk_assessment",
    "biodiversity", "biogeography", "statistics", "systematic_review",
    "meta_analysis", "protocol", "method_comparison", "process_optimization",
    "industrial_application", "bioprocessing", "biofuel_feedstock",
]

def get_system_prompt():
    cats = ", ".join(f'"{c}"' for c in SUGGESTED_CATEGORIES)
    return f"""You are an expert bioinformatician and data annotator.

Classify the scientific text into ONE category.

Try first from this list:
[{cats}]

If none fits, invent a new short category name (lowercase_underscore) and provide only the keywords that determined this classification.

Output raw JSON only:

If exists in list:
{{"category": "existing_category", "is_new": false}}

If new:
{{"category": "new_category_name", "is_new": true, "keywords": ["kw1", "kw2", ...]}}"""

def build_text(record):
    t = record.get('type', '')
    
    if t in ('conceptual', 'factual'):
        q = record.get('question', '') or ''
        a = record.get('answer', '')   or ''
        return f"{q} {a}".strip()
    
    else:  
        i = record.get('instruction', '') or ''
        o = record.get('output', '')      or ''
        return f"{i} {o}".strip()

def classify_with_llm(idx, record):
    text = build_text(record)
    if not text:
        return idx, None, False, []
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user",   "content": f"TEXT:\n{text[:1500]}"},
            ],
            temperature=0.2,
        )
        raw   = response.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            parsed   = json.loads(match.group(0))
            cat      = parsed.get('category', '').lower().replace(" ", "_")
            is_new   = parsed.get('is_new', False)
            keywords = parsed.get('keywords', [])
            return idx, cat, is_new, keywords
    except Exception as e:
        print(f"  [ERROR] idx={idx} {type(e).__name__}: {e}", flush=True)
    return idx, None, False, []

def save_jsonl(rows, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Fichier introuvable : {INPUT_FILE}")
        return

    rows = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    print(f"{len(rows):,} enregistrements  |  workers={MAX_WORKERS}")

    new_categories_found = {}
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(classify_with_llm, i, row): i for i, row in enumerate(rows)}

        with tqdm(total=len(rows)) as pbar:
            for future in as_completed(futures):
                idx, cat, is_new, keywords = future.result()

                if cat:
                    rows[idx]['category'] = cat
                    if is_new:
                        rows[idx]['category_keywords'] = keywords
                        new_categories_found[cat] = keywords
                        print(f"  [NEW]  idx={idx} → {cat}", flush=True)
                    else:
                        print(f"  [OK]   idx={idx} → {cat}", flush=True)
                else:
                    print(f"  [FAIL] idx={idx}", flush=True)

                processed += 1
                pbar.update(1)

                if processed % AUTOSAVE_INTERVAL == 0:
                    save_jsonl(rows, OUTPUT_FILE)

    save_jsonl(rows, OUTPUT_FILE)
    print(f"\n✅ {OUTPUT_FILE}  —  {processed:,} traités")

    if new_categories_found:
        print(f"\n{len(new_categories_found)} nouvelles catégories :")
        for cat, kws in new_categories_found.items():
            print(f"   {cat:<35} → {kws}")

if __name__ == "__main__":
    main()