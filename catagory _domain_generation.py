import os
import json
import re
import sys
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="ollama"  
)
MODEL_NAME =  "gpt-oss:20b"


INPUT_FILE = "combined_cleaned_dataset.jsonl" 
OUTPUT_FILE = "dataset_final_enriched.jsonl"
AUTOSAVE_INTERVAL = 200
MAX_WORKERS = 8  # Nombre d'appels LLM en parallèle

SUGGESTED_CATEGORIES = [
    "alignment", "annotation", "assembly", "assembly_qc", "bin_qc",
    "binning", "diversity_analysis", "functional_profiling", "genomics_infra",
    "host_decontamination", "machine_learning", "multiomics", "pipeline_design",
    "qc_preprocessing", "quantification", "sequencing", "statistical_analysis",
    "taxonomy", "visualization"
]

SUGGESTED_DOMAINS = [
    "QC & Preprocessing", "Assembly", "Binning", "Annotation",
    "Taxonomy / Classification", "Metagenomics Tools", "Errors & Debugging",
    "Performance & Scaling", "Gut Microbiota", "Metabolites / SCFA",
    "Fermentation", "Diet / Nutrition", "Poultry Biology",
    "Energy / Metabolism", "Epigenetics", "Gut Barrier / Mucosal",
    "Immunology", "Receptor Signaling"
]

def get_system_prompt():
    return f"""You are an expert bioinformatician and biologist data annotator.
Your task is to classify a given scientific text into EXACTLY ONE Category and EXACTLY ONE primary Domain.

Here are some SUGGESTED Categories: {SUGGESTED_CATEGORIES}
Here are some SUGGESTED Domains: {SUGGESTED_DOMAINS}

RULES:
1. You may choose from the suggested lists, OR INVENT a new short, descriptive Category/Domain if needed.
2. Output your response STRICTLY as a raw JSON object. NO explanation, NO markdown.

EXPECTED JSON FORMAT:
{{
    "category": "your_category",
    "domain": "Your Domain"
}}"""


def classify_with_llm(text):
    try:
        truncated_text = text[:600] if isinstance(text, str) else str(text)[:600]
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": f"Classify:\n{truncated_text}"}
            ],
            temperature=0.3,
            max_tokens=100,
        )
        
        raw_output = response.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            parsed_data = json.loads(match.group(0))
            
            cat = parsed_data.get('category', 'other')
            domain = parsed_data.get('domain', 'Other / Unclassified')
            
            if isinstance(cat, str): cat = cat.lower().replace(" ", "_")
            if isinstance(domain, str): domain = domain.title()
                
            return cat, domain
        else:
            print(f"  [NO-JSON] {raw_output[:200]}", flush=True)
            
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}", flush=True)
    return None, None


def is_invalid_category(cat):
    return cat in ('other', 'Other', None, '') or not cat


def is_invalid_domain(domains):
    if not isinstance(domains, list):
        return True
    if len(domains) == 0:
        return True
    if 'Other / Unclassified' in domains:
        return True
    return False


def extract_text(row):
    parts = []
    for key in ['instruction', 'output', 'question', 'answer', 'steps', 'constraints']:
        val = row.get(key)
        if val is not None and str(val).strip():
            if isinstance(val, list):
                parts.append(" ".join(map(str, val)))
            else:
                parts.append(str(val))
    return " ".join(parts).strip()


def process_one(idx, row):
    """Traite une seule ligne — appelée en parallèle."""
    text = extract_text(row)
    if not text:
        return idx, None, None
    new_cat, new_domain = classify_with_llm(text)
    return idx, new_cat, new_domain


def save_jsonl(rows, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():

    if not os.path.exists(INPUT_FILE):
        print("File introuvable.")
        return

    # 1. Lire le dataset (format hétérogène préservé)
    rows = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # 2. Identifier les lignes à traiter
    target_indices = []
    for i, row in enumerate(rows):
        cat = row.get('category', '')
        domains = row.get('domains', [])
        if is_invalid_category(cat) or is_invalid_domain(domains):
            target_indices.append(i)

    print(f"{len(target_indices)} / {len(rows)} lignes a traiter.")
    print(f"Workers paralleles: {MAX_WORKERS}")

    # 3. Traitement parallèle
    processed_count = 0
    ok_count = 0
    fail_count = 0

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}

            # Soumettre toutes les tâches
            for idx in target_indices:
                future = executor.submit(process_one, idx, rows[idx])
                futures[future] = idx

            # Collecter les résultats avec barre de progression
            pbar = tqdm(total=len(target_indices))
            for future in as_completed(futures):
                idx, new_cat, new_domain = future.result()
                row = rows[idx]

                if new_cat is not None and new_domain is not None:
                    if is_invalid_category(row.get('category', '')):
                        row['category'] = new_cat
                    if is_invalid_domain(row.get('domains', [])):
                        row['domains'] = [new_domain]
                    ok_count += 1
                else:
                    fail_count += 1

                processed_count += 1
                pbar.update(1)
                pbar.set_postfix(ok=ok_count, fail=fail_count)

                # Autosave
                if processed_count % AUTOSAVE_INTERVAL == 0:
                    save_jsonl(rows, OUTPUT_FILE)

            pbar.close()

    except KeyboardInterrupt:
        print("\nInterrompu.")
    finally:
        save_jsonl(rows, OUTPUT_FILE)
        print(f"\nTermine: {ok_count} OK, {fail_count} FAIL sur {processed_count} traites.")
        print(f"Fichier genere : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
