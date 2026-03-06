import os
import sys
import json
import re
import signal
import atexit
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_BASE   = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
API_KEY    = os.getenv("LLM_API_KEY", "ollama")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-oss:20b")

client = OpenAI(base_url=API_BASE, api_key=API_KEY, timeout=120)

INPUT_FILE        = "paper_other.jsonl"
OUTPUT_FILE       = "paper_other_classified.jsonl"
MAX_WORKERS       = 2
AUTOSAVE_INTERVAL = 100

_rows_ref           = None
_output_file_ref    = None
_shutdown_requested = False

def _emergency_save():
    if _rows_ref is not None and _output_file_ref is not None:
        try:
            save_jsonl(_rows_ref, _output_file_ref)
            done = sum(1 for r in _rows_ref if r.get('category', 'other') not in ('other', '', None))
            print(f"\nSauvegarde : {done}/{len(_rows_ref)} classifiés → {_output_file_ref}", flush=True)
        except Exception as e:
            print(f"\nÉchec sauvegarde : {e}", flush=True)

def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    sys.exit(0)

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
atexit.register(_emergency_save)

SUGGESTED_CATEGORIES = {
    "pipeline_design": [
        "pipeline", "workflow", "steps", "orchestration",
        "snakemake", "nextflow", "cwl", "workflow management"
    ],
    "qc_preprocessing": [
        "quality control", "qc", "fastqc", "multiqc", "adapter trimming",
        "quality trimming", "filtering", "low quality reads", "cutadapt",
        "trimmomatic", "bbduk", "adapter removal", "deduplication",
        "demultiplexing preprocessing", "barcode removal", "primer removal",
        "chimera removal", "read merging", "host removal preprocessing",
        "format conversion", "preprocessing pipeline", "quality control sequencing",
        "quality filtering reads", "Phred score", "duplicate removal",
        "contamination removal QC", "read filtering QC"
    ],
    "sequencing": [
        "sequencing", "fastq", "illumina", "nanopore", "pacbio", "16s",
        "amplicon", "shotgun", "paired-end", "single-end", "library preparation",
        "16S rRNA amplicon", "ITS amplicon", "V3-V4 region", "DADA2", "QIIME2",
        "ASV calling", "primer pair", "amplicon bioinformatics", "OTU clustering",
        "amplicon read", "targeted sequencing", "Illumina sequencing",
        "nanopore sequencing", "PacBio sequencing", "sequencing depth",
        "paired-end sequencing", "single-end sequencing", "amplicon sequencing",
        "shotgun sequencing", "RNA-seq", "sequencing library", "sequencing error",
        "base quality score", "sequencing platform"
    ],
    "host_decontamination": [
        "host removal", "decontamination", "host filtering", "human contamination",
        "bowtie2 host", "kneaddata", "bmtool", "bbmap", "host read removal",
        "human genome decontamination", "Bowtie2 host subtraction", "BMTagger",
        "Kraken2 host filter", "mouse genome removal", "host genome filtering",
        "non-host reads", "microbial read enrichment", "host contamination removal",
        "depletion strategy", "host-associated metagenomics", "decontamination efficiency"
    ],
    "alignment": [
        "alignment", "reference alignment", "mapping", "bam", "sam",
        "bowtie", "bowtie2", "bwa", "minimap2", "reference genome",
        "HISAT2", "SAM file", "BAM file", "CIGAR string", "mapping rate",
        "aligned reads", "STAR aligner", "coverage depth"
    ],
    "assembly": [
        "assembly", "contigs", "scaffolds", "de novo assembly", "co-assembly",
        "megahit", "metaspades", "spades", "idba-ud", "contig", "scaffold",
        "N50 statistic", "SPAdes assembler", "metagenome assembly",
        "genome reconstruction", "assembly quality metric", "k-mer assembly", "read overlap"
    ],
    "assembly_qc": [
        "assembly quality", "n50", "l50", "contig length", "assembly statistics",
        "quast", "metaquast", "checkm", "assembly completeness", "QUAST evaluation",
        "N50 score", "L50 metric", "contig length distribution", "assembly fragmentation",
        "misassembly detection", "genome fraction covered", "assembly evaluation",
        "reference-based QC", "BUSCO completeness", "assembly benchmark"
    ],
    "binning": [
        "binning", "metagenome bins", "mag", "metagenome-assembled genomes",
        "metabat", "maxbin", "concoct", "das tool", "metagenome binning",
        "MAG recovery", "MetaBAT2", "MaxBin2", "CONCOCT binning", "bin refinement",
        "bin quality control", "CheckM", "tetranucleotide frequency",
        "coverage-based binning", "differential coverage", "co-assembly binning"
    ],
    "bin_qc": [
        "bin quality", "completeness", "contamination", "checkm", "gtdb-tk",
        "mag quality", "bin completeness", "bin contamination", "CheckM quality",
        "MAG quality assessment", "genome completeness score", "single copy marker gene",
        "bin quality filter", "completeness threshold", "contamination threshold",
        "bin statistics", "high quality MAG", "medium quality MAG", "bin refinement score"
    ],
    "taxonomy": [
        "taxonomy", "taxonomic profiling", "otu", "asv", "species abundance",
        "kraken", "kraken2", "bracken", "metaphlan", "centrifuge", "gtdb",
        "taxonomic classification", "16S rRNA taxonomy", "OTU ASV", "SILVA taxonomy",
        "NCBI taxonomy", "species identification", "phylum genus family",
        "relative abundance taxonomy", "taxonomic diversity", "taxonomic assignment"
    ],
    "annotation": [
        "annotation", "functional annotation", "gene prediction", "orfs", "cds",
        "kegg", "eggnog", "cog", "pfam", "interpro", "prokka", "dram",
        "gene annotation", "genome annotation", "AUGUSTUS", "GO terms",
        "protein domain", "Pfam database", "BLAST annotation", "CDS prediction",
        "structural annotation", "predicted gene"
    ],
    "functional_profiling": [
        "pathway analysis", "functional profiling", "metabolic pathways",
        "enzyme abundance", "humann", "humann3", "minpath"
    ],
    "quantification": [
        "abundance", "counts", "normalization", "relative abundance",
        "coverage", "rpkm", "tpm", "fpkm", "depth"
    ],
    "diversity_analysis": [
        "alpha diversity", "beta diversity", "shannon", "simpson",
        "bray curtis", "ordination", "pcoa", "nmds"
    ],
    "statistical_analysis": [
        "differential abundance", "statistical testing", "significance",
        "anova", "wilcoxon", "lefse", "deseq2", "aldex2"
    ],
    "visualization": [
        "visualization", "plot", "heatmap", "barplot", "boxplot",
        "ordination plot", "phyloseq", "ggplot", "ggtree", "PCA plot",
        "scatter plot", "volcano plot", "phylogenetic tree visualization",
        "ggplot2", "matplotlib", "R visualization microbiome",
        "abundance plot", "interactive visualization"
    ],
    "machine_learning_metagenomics": [
        "machine learning", "deep learning", "classification", "prediction",
        "random forest", "svm", "neural network", "ML taxonomic classification",
        "ML read classification", "ML contig classification", "ML binning",
        "deep learning binning", "microbiome prediction model",
        "random forest microbiome", "SVM microbiome", "neural network metagenomics",
        "CNN metagenomics", "transformer genomics", "graph neural network metagenomics",
        "embedding metagenomics", "k-mer embedding", "feature engineering microbiome",
        "supervised microbiome learning", "unsupervised microbiome learning",
        "cross-validation microbiome", "model evaluation microbiome",
        "feature importance microbiome", "predictive microbiome model"
    ],
    "multiomics": [
        "multi-omics", "integration", "metabolomics", "proteomics",
        "transcriptomics", "systems biology", "metagenomics integration",
        "metatranscriptomics", "metaproteomics", "metabolomics integration",
        "multi-omics data integration", "omics layer", "cross-omics correlation",
        "microbiome multi-omics", "host-microbiome omics",
        "integrated multi-omics analysis", "omics dataset combination",
        "multi-omics pipeline", "systems-level omics"
    ],
    "genomics_infra": [
        "container", "docker", "singularity", "conda", "environment",
        "genomic", "reference genome", "versioning", "reproducibility",
        "HPC cluster", "cluster computing genomics", "memory usage",
        "CPU threads", "parallel processing", "SLURM scheduler",
        "cloud computing genomics", "storage genomics", "scalability genomics",
        "runtime optimization", "bioinformatics infrastructure",
        "Docker Singularity", "job scheduler"
    ],
    "association_analysis": [
        "microbiome association study", "GWAS microbiome", "host-microbiome association",
        "QTL mapping", "correlation analysis", "trait-microbiome link",
        "linear mixed model", "MaAsLin2", "multivariate association",
        "covariate adjustment", "confounding control", "phenotype association", "taxa association"
    ],
    "dna_extraction": [
        "DNA extraction", "nucleic acid extraction", "lysis buffer",
        "phenol-chloroform extraction", "extraction kit", "PowerSoil kit",
        "MoBio", "DNA yield", "DNA purity", "A260/A280 ratio",
        "DNA quality check", "low biomass extraction", "DNA extraction protocol",
        "RNA extraction", "extraction efficiency", "extraction kit comparison",
        "bead beating", "chemical extraction", "enzymatic extraction",
        "extraction yield", "extraction purity", "extraction protocol"
    ],
    "bioinformatic_algorithm_optimization": [
        "algorithm optimization", "heuristic method", "genetic algorithm",
        "threshold", "k-mer size", "word size", "window size", "seed length",
        "e-value", "minimum coverage", "scoring matrix", "default parameter"
    ],
    "errors_&_debugging": [
        "pipeline error", "debugging bioinformatics", "error message",
        "crash log", "tool error", "memory error", "format error",
        "dependency error", "permission error", "segmentation fault",
        "runtime error", "unexpected output", "debugging strategy"
    ],
    "experiment_metadata": [
        "experiment metadata", "experimental annotation", "sample annotation",
        "run metadata", "protocol metadata", "condition metadata",
        "SRA metadata", "MIxS metadata", "MIMARKS", "experimental variable",
        "metadata completeness", "metadata standard", "study metadata"
    ],
    "reference_database_usage": [
        "database information query", "NCBI query", "sequence database search",
        "genomic information request", "data retrieval bioinformatics",
        "metadata information query", "reference database query",
        "functional database query", "taxon information query",
        "gene information query", "pathway information query",
        "protein information query", "accession number query",
        "BLAST search", "genomic search", "GTDB query",
        "KEGG lookup", "SRA metadata"
    ],
}

# ── calculé une seule fois au démarrage ───────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert bioinformatician and data annotator.\n\n"
    "Classify the scientific text into ONE category using the list below.\n"
    "Each category is shown with sample keywords to help you decide.\n\n"
    + "\n".join(f'  "{k}": {json.dumps(v[:9])}' for k, v in SUGGESTED_CATEGORIES.items())
    + "\n\nIf none fits, invent a new short category name (lowercase_underscore) "
    "and provide only the keywords that determined this classification.\n\n"
    "Output raw JSON only, no explanation:\n\n"
    'If exists in list:\n{"category": "existing_category", "is_new": false}\n\n'
    'If new:\n{"category": "new_category_name", "is_new": true, "keywords": ["kw1", "kw2", ...]}'
)


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
                {"role": "system", "content": SYSTEM_PROMPT},
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


def verify_workers(n_workers):
    print(f"\n--- Verification parallelisation ({n_workers} workers) ---")
    active_threads = []
    lock = threading.Lock()
    max_concurrent = [0]

    def _worker_task(wid):
        with lock:
            active_threads.append(wid)
            if len(active_threads) > max_concurrent[0]:
                max_concurrent[0] = len(active_threads)
        time.sleep(0.5)
        with lock:
            active_threads.remove(wid)
        return wid

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker_task, i): i for i in range(n_workers)}
        results = [f.result() for f in as_completed(futs)]
    elapsed = time.time() - t0

    parallel = max_concurrent[0] >= min(n_workers, 2)
    seq_time = 0.5 * n_workers

    if parallel and elapsed < seq_time * 0.8:
        print(f"  [PASS] {max_concurrent[0]} workers simultanes | {elapsed:.2f}s (seq aurait pris ~{seq_time:.1f}s)")
    else:
        print(f"  [WARN] max concurrent={max_concurrent[0]} | {elapsed:.2f}s — parallelisation limitee")
    print()
    return parallel


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Fichier introuvable : {INPUT_FILE}")
        return

    verify_workers(MAX_WORKERS)

    if os.path.exists(OUTPUT_FILE):
        rows = []
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        already_done = {i for i, r in enumerate(rows) if r.get('category', 'other') not in ('other', '', None)}
        print(f"Reprise : {len(already_done)}/{len(rows)} déjà classifiés")
    else:
        rows = []
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        already_done = set()

    to_process = [(i, row) for i, row in enumerate(rows) if i not in already_done]
    print(f"{len(rows):,} total  |  {len(to_process):,} à traiter  |  workers={MAX_WORKERS}")

    if not to_process:
        print("Tout est déjà classifié !")
        return

    new_categories_found = {}
    processed = 0

    global _rows_ref, _output_file_ref
    _rows_ref = rows
    _output_file_ref = OUTPUT_FILE

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(classify_with_llm, i, row): i for i, row in to_process}

            with tqdm(total=len(to_process)) as pbar:
                for future in as_completed(futures):
                    if _shutdown_requested:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

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
                        print(f"  [SAVE] autosave @ {processed}", flush=True)

    except (KeyboardInterrupt, SystemExit):
        print(f"\nInterruption détectée après {processed} traités", flush=True)
    except Exception as e:
        print(f"\nErreur inattendue : {type(e).__name__}: {e}", flush=True)
    finally:
        save_jsonl(rows, OUTPUT_FILE)
        done = sum(1 for r in rows if r.get('category', 'other') not in ('other', '', None))
        print(f"\nSauvegarde finale : {done}/{len(rows)} classifiés → {OUTPUT_FILE}", flush=True)
        _rows_ref = None

    if new_categories_found:
        print(f"\n{len(new_categories_found)} nouvelles catégories :")
        for cat, kws in new_categories_found.items():
            print(f"   {cat:<35} → {kws}")


if __name__ == "__main__":
    main()