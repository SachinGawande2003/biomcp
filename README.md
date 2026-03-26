# 🧬 BioMCP — Bioinformatics Model Context Protocol Server

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io)
[![Tools: 27](https://img.shields.io/badge/Tools-27-blue.svg)](#tools-27-total)
[![Databases: 15+](https://img.shields.io/badge/Databases-15+-purple.svg)](#databases)

**The most comprehensive Model Context Protocol server for life sciences.**  
Connect Claude to every major biological database and state-of-the-art AI models — no API wrappers, no copy-pasting, just science.

[🚀 Quick Start](#quick-start) • [🔧 Tools](#tools-27-total) • [📊 Databases](#databases) • [💡 Examples](#usage-examples) • [🤝 Contributing](#contributing)

</div>

---

## What is BioMCP?

BioMCP bridges Claude and the world's life sciences databases through the [Model Context Protocol](https://modelcontextprotocol.io). Ask Claude to **search PubMed**, **predict protein structure** with AI, **generate DNA sequences**, **find drug targets**, **query clinical trials**, **analyze single-cell data**, or **generate research hypotheses** — all in natural language, all in real time.

```
You → "What drugs target EGFR and what clinical trials are recruiting?"
Claude + BioMCP → Queries ChEMBL + ClinicalTrials.gov simultaneously → Structured answer
```

---

## Tools (27 total)

### 📚 Literature & NCBI
| Tool | Description |
|------|-------------|
| `search_pubmed` | Full PubMed search with MeSH, Boolean, field tags. Returns abstracts, authors, DOIs |
| `get_gene_info` | NCBI Gene: symbol, location, aliases, RefSeq IDs, functional summary |
| `run_blast` | NCBI BLAST alignment (blastp/blastn/blastx/tblastn) — async, non-blocking |

### 🧪 Proteins & Structures
| Tool | Description |
|------|-------------|
| `get_protein_info` | UniProt Swiss-Prot: function, domains, PTMs, GO terms, disease links |
| `search_proteins` | Search UniProt by gene, function, or disease with species filtering |
| `get_alphafold_structure` | AlphaFold DB: pLDDT confidence scores, PDB/mmCIF download URLs, PAE data |
| `search_pdb_structures` | RCSB PDB: experimental structures with resolution, method, deposition date |

### 🔬 Pathways
| Tool | Description |
|------|-------------|
| `search_pathways` | KEGG pathway search with organism-specific viewer links |
| `get_pathway_genes` | All genes in a KEGG pathway with descriptions |
| `get_reactome_pathways` | Reactome pathways for a gene with hierarchy and diagram links |

### 💊 Drug Discovery
| Tool | Description |
|------|-------------|
| `get_drug_targets` | ChEMBL: drugs/compounds targeting a gene with IC50, Ki, approval status |
| `get_compound_info` | ChEMBL compound details: SMILES, ADMET, Lipinski Ro5, QED, indications |
| `get_gene_disease_associations` | Open Targets: gene-disease evidence across genetics, drugs, and pathways |

### 🧬 Genomics & Expression
| Tool | Description |
|------|-------------|
| `get_gene_variants` | Ensembl variants: SNPs, indels, consequence types, clinical significance |
| `search_gene_expression` | NCBI GEO datasets for a gene with organism, platform, sample counts |
| `search_scrna_datasets` | Human Cell Atlas single-cell RNA-seq by tissue and technology |

### 🏥 Clinical
| Tool | Description |
|------|-------------|
| `search_clinical_trials` | ClinicalTrials.gov: trials with status, phase, interventions, eligibility |
| `get_trial_details` | Full trial protocol: arms, outcomes, contacts |

### 🤖 AI-Powered (NVIDIA NIM)
| Tool | Description |
|------|-------------|
| `predict_structure_boltz2` | MIT Boltz-2: Protein/DNA/RNA/ligand structure prediction + binding affinity (FEP accuracy, 1000x faster) |
| `generate_dna_evo2` | Arc Evo2-40B: Generate DNA sequences with 40B parameter genomic foundation model |
| `score_sequence_evo2` | Compare wildtype vs variant DNA sequences for variant effect prediction |
| `design_protein_ligand` | Full drug-discovery pipeline: UniProt → Boltz-2 structure + affinity in one call |

### 🌐 Integrated & Advanced
| Tool | Description |
|------|-------------|
| `multi_omics_gene_report` | **Flagship**: 7+ databases queried in parallel → one integrated gene report |
| `query_neuroimaging_datasets` | OpenNeuro + NeuroVault: fMRI/EEG/MEG datasets with acquisition metadata |
| `generate_research_hypothesis` | Literature mining → data-driven testable hypotheses with supporting evidence |

---

## Databases & AI Models

| Source | Domain | URL |
|--------|--------|-----|
| **Traditional Databases** |
| NCBI PubMed | Literature | https://pubmed.ncbi.nlm.nih.gov |
| NCBI Gene | Genomics | https://www.ncbi.nlm.nih.gov/gene |
| NCBI BLAST | Sequence Alignment | https://blast.ncbi.nlm.nih.gov |
| NCBI GEO | Gene Expression | https://www.ncbi.nlm.nih.gov/geo |
| UniProt Swiss-Prot | Proteomics | https://www.uniprot.org |
| AlphaFold DB | Protein Structure | https://alphafold.ebi.ac.uk |
| RCSB PDB | Protein Structure | https://www.rcsb.org |
| KEGG | Pathways | https://www.kegg.jp |
| Reactome | Pathways | https://reactome.org |
| ChEMBL | Drug Discovery | https://www.ebi.ac.uk/chembl |
| Open Targets | Gene-Disease | https://platform.opentargets.org |
| Ensembl | Genomics | https://www.ensembl.org |
| ClinicalTrials.gov | Clinical | https://clinicaltrials.gov |
| Human Cell Atlas | Single-Cell | https://data.humancellatlas.org |
| OpenNeuro | Neuroimaging | https://openneuro.org |
| NeuroVault | Neuroimaging | https://neurovault.org |
| **AI Models (NVIDIA NIM)** |
| MIT Boltz-2 | Structure Prediction | https://build.nvidia.com/mit/boltz2 |
| Arc Evo2-40B | DNA Generation | https://build.nvidia.com/arc/evo2-40b |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Claude Desktop or any MCP-compatible client
- (Optional) NCBI API key for higher rate limits
- (Optional) NVIDIA API keys for AI tools

### Installation

```bash
# Clone the repository
git clone https://github.com/SachinGawande2003/biomcp.git
cd biomcp

# Install (standard)
pip install -e .

# Install with neuroimaging support
pip install -e ".[neuroimaging]"

# Install with dev dependencies
pip install -e ".[dev]"
```

### Configure Claude Desktop

Add to your `claude_desktop_config.json`:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "biomcp": {
      "command": "biomcp",
      "env": {
        "NCBI_API_KEY": "your_ncbi_api_key_here"
      }
    }
  }
}
```

> **💡 Tip**: Get a free [NCBI API key](https://www.ncbi.nlm.nih.gov/account/) to increase rate limits from 3 to 10 requests/second.
>
> **🚀 New**: Get free NVIDIA API keys for AI tools at [build.nvidia.com/mit/boltz2](https://build.nvidia.com/mit/boltz2) and [build.nvidia.com/arc/evo2-40b](https://build.nvidia.com/arc/evo2-40b).

### Restart Claude Desktop and test:

```
"Search PubMed for recent papers on CAR-T cell therapy in B-cell lymphoma"
"Get the AlphaFold structure for TP53 and tell me about the confidence scores"
"What drugs are approved that target EGFR?"
"Generate a multi-omics report for KRAS"
"Predict the structure of EGFR with ligand CC1=CC=CC=C1 and compute binding affinity"
"Generate a DNA sequence starting with ATGGCG..."
```

---

## Usage Examples

### Literature Mining
```
"Search PubMed for BRCA1 CRISPR correction methods published in the last 2 years"
"Find review articles about PD-1/PD-L1 immune checkpoint inhibitors"
```

### Protein Analysis
```
"Get UniProt info for human TP53 (P04637) including its domains and disease associations"
"Search for AlphaFold structures for insulin receptor"
"Find all PDB crystal structures of BRAF kinase domain resolved below 2.5 Ångström"
```

### Drug Discovery
```
"What are the top ChEMBL compounds targeting KRAS G12C mutation?"
"Get compound info for imatinib (CHEMBL941)"
"Show me gene-disease associations for BRCA1 with evidence scores"
```

### AI-Powered Structure Prediction
```
"Predict the 3D structure of insulin (sequence: ...) with ligand CCO"
"Compute binding affinity between EGFR and gefitinib (SMILES: ...)"
"Get structure prediction for a protein-protein complex"
```

### AI-Powered DNA Generation
```
"Generate a 200bp promoter sequence starting with ATG"
"Compare wildtype vs variant DNA sequence for TP53 mutation"
"Generate regulatory element for gene expression"
```

### Multi-Omics Report (Flagship)
```
"Generate a complete multi-omics report for EGFR"
```
This single command queries **7 databases in parallel** and returns:
- Genomic location and gene summary (NCBI Gene)
- Recent publications (PubMed)  
- Protein function and structure (UniProt + AlphaFold)
- Biological pathways (Reactome)
- Drug targets and clinical compounds (ChEMBL)
- Disease associations with scores (Open Targets)
- Expression datasets (GEO)
- Active clinical trials (ClinicalTrials.gov)

### Clinical Research
```
"Find Phase 2 recruiting trials for KRAS-mutant non-small cell lung cancer"
"Get full details for clinical trial NCT04280705"
```

### Neuroimaging
```
"Find fMRI datasets for hippocampus in Alzheimer's disease"
"Search for EEG datasets studying working memory in prefrontal cortex"
```

---

## Architecture

```
biomcp/
├── src/biomcp/
│   ├── server.py          # MCP server — tool registry & dispatcher
│   ├── tools/
│   │   ├── ncbi.py        # PubMed, Gene, BLAST
│   │   ├── proteins.py    # UniProt, AlphaFold, PDB
│   │   ├── pathways.py    # KEGG, Reactome, ChEMBL, Open Targets
│   │   ├── advanced.py    # ClinicalTrials, GEO, scRNA, Ensembl,
│   │   │                  # Multi-Omics, Neuroimaging, Hypothesis
│   │   └── nvidia_nim.py  # Boltz-2, Evo2-40B AI models
│   └── utils/
│       └── __init__.py    # Rate limiter, cache, validators, HTTP client
├── tests/
├── pyproject.toml
└── README.md
```

**Key design decisions:**
- **Async-first**: All API calls are fully async with `httpx`, never blocking
- **Rate limiting**: Token-bucket limiter per service respects each API's limits
- **Smart caching**: TTL-based per-namespace cache (1h literature, 7d structures)  
- **Retry logic**: Exponential backoff via `tenacity` for transient failures
- **Validation**: Input validation before any network call — never wastes API quota

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NCBI_API_KEY` | NCBI API key (increases rate limit to 10/s) | None (3/s) |
| `NVIDIA_BOLTZ2_API_KEY` | NVIDIA API key for Boltz-2 structure prediction | None |
| `NVIDIA_EVO2_API_KEY` | NVIDIA API key for Evo2-40B DNA generation | None |
| `BIOMCP_LOG_LEVEL` | Log level: DEBUG/INFO/WARNING/ERROR | INFO |

Get free API keys:
- **NCBI**: https://www.ncbi.nlm.nih.gov/account/
- **NVIDIA Boltz-2**: https://build.nvidia.com/mit/boltz2
- **NVIDIA Evo2-40B**: https://build.nvidia.com/arc/evo2-40b

---

## Contributing

Contributions are welcome! Whether it's adding a new database, fixing a bug, improving documentation, or integrating new AI models.

```bash
# Development setup
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=biomcp

# Lint + type check
ruff check src/
mypy src/
```

### Ideas for contributions
- Add OMIM (genetic disease database) integration
- Integrate STRING protein interaction network
- Add PanCancer Atlas data access
- Improve scRNA-seq analysis beyond dataset discovery
- Add GTEx gene expression by tissue
- Integrate DisGeNET for disease-gene associations

---

## Citation

If you use BioMCP in your research, please cite:

```bibtex
@software{biomcp2025,
  title   = {BioMCP: A Comprehensive MCP Server for Bioinformatics, AI Models, and Life Sciences},
  year    = {2025},
  url     = {https://github.com/SachinGawande2003/biomcp},
  license = {MIT}
}
```

---

## License

MIT License — free for academic and commercial use.

---

<div align="center">
Built for researchers, by researchers. 🔬<br>
<b>Star ⭐ this repo if BioMCP helps your science!</b>
</div>