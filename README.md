# 🧬 Heuris-BioMCP — Bioinformatics Model Context Protocol Server

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io)
[![Tools: 31](https://img.shields.io/badge/Tools-31-blue.svg)](#tools-31-curated-public-surface)
[![Databases: 30+](https://img.shields.io/badge/Databases-30+-purple.svg)](#databases--ai-models)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://heuris-biomcp.onrender.com)

**Strategic Model Context Protocol server for life sciences.**  
Connect ChatGPT, Claude, and other MCP clients to a curated biology tool surface built for research, translational workflows, and production review.

[🚀 Quick Start](#quick-start) • [🔧 Tools](#tools-31-curated-public-surface) • [📊 Databases](#databases--ai-models) • [💡 Examples](#usage-examples) • [🤝 Contributing](#contributing)

</div>

---

## Live Demo

Try Heuris-BioMCP without installing — connect to our live server:

```
https://heuris-biomcp.onrender.com/mcp
```

### Connect to Live Server

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "heuris-biomcp": {
      "url": "https://heuris-biomcp.onrender.com/mcp"
    }
  }
}
```

> **Note**: `https://heuris-biomcp.onrender.com/mcp` is the recommended remote MCP endpoint for modern clients. The legacy SSE endpoint remains available at `https://heuris-biomcp.onrender.com/sse`.
>
> **Claude update**: Anthropic now routes remote MCP connections through `Customize > Connectors`. Direct remote URLs in `claude_desktop_config.json` are no longer the recommended Claude Desktop path.

---

## See Heuris-BioMCP in Action

![Heuris-BioMCP Demo](assets/biomcp-demo.gif)

### Quick Demo Video

Watch how to connect Heuris-BioMCP to Claude Desktop:

<a href="https://www.youtube.com/watch?v=YOUR_VIDEO_ID">
  <img src="assets/youtube-thumbnail.png" alt="Watch Demo" width="400"/>
</a>

> **Tip**: Coming soon - video walkthrough of connecting Heuris-BioMCP and running your first query!

---

## What is Heuris-BioMCP?

Heuris-BioMCP bridges MCP clients and high-value life-science data sources through a curated public tool surface. The server exposes the workflows that matter most for product review and research use, while lower-level helper modules remain internal for composition and planner logic.

```
You -> "What drugs target EGFR and what clinical trials are recruiting?"
ChatGPT + Heuris-BioMCP -> Queries ChEMBL + ClinicalTrials.gov -> Structured answer
```

---

## What's New in v2.3

- Curated the public MCP surface from a broad 71-tool registry down to a strategy-driven set of 31 review-friendly tools
- Merged operational suites into workflow tools: `find_protein`, `pathway_analysis`, `crispr_analysis`, `drug_safety`, `variant_analysis`, and `session`
- Added new translational tools: `drug_interaction_checker`, `protein_binding_pocket`, `biomarker_panel_design`, `pharmacogenomics_report`, `protein_family_analysis`, `network_enrichment`, `rnaseq_deconvolution`, `structural_similarity`, `rare_disease_diagnosis`, and `genome_browser_snapshot`
- Removed low-signal or niche tools from the public MCP registry while keeping lower-level code available internally
- Preserved hosted HTTP/SSE deployment and operational endpoints for production-style MCP review

---

## Tools (31 curated public surface)

### Core Research
| Tool | Description |
|------|-------------|
| `search_pubmed` | PubMed literature search with MeSH, Boolean syntax, abstracts, and metadata |
| `get_gene_info` | NCBI Gene summary with aliases, locus, and functional context |
| `run_blast` | NCBI BLAST sequence alignment |
| `get_protein_info` | Full UniProt Swiss-Prot protein record |
| `find_protein` | Unified UniProt plus PDB protein discovery workflow |
| `get_alphafold_structure` | AlphaFold structure metadata and confidence summary |
| `pathway_analysis` | Merged KEGG plus Reactome pathway workflow |
| `get_drug_targets` | ChEMBL drug-target evidence for a gene |
| `get_gene_disease_associations` | Open Targets translational gene-disease evidence |
| `search_clinical_trials` | ClinicalTrials.gov recruiting-trial search |
| `multi_omics_gene_report` | Integrated multi-database flagship gene report |

### AI And Engineering Workflows
| Tool | Description |
|------|-------------|
| `predict_structure_boltz2` | Boltz-2 structure workflow with optional protein-ligand mode |
| `generate_dna_evo2` | Evo2 generation or WT-vs-variant scoring workflow |
| `crispr_analysis` | Merged CRISPR design, scoring, off-target, base-edit, and repair workflow |
| `drug_safety` | Merged FDA safety workflow for events, signals, labels, and comparisons |
| `variant_analysis` | Merged ACMG, gnomAD, ClinVar, splice, and integrated variant reporting |
| `session` | Merged entity, graph, export, and adaptive planning workflow |

### High-Value Translational Tools
| Tool | Description |
|------|-------------|
| `find_repurposing_candidates` | Drug repurposing workflow over literature, trials, and target evidence |
| `verify_biological_claim` | Cross-database biological claim verification |
| `search_cbio_mutations` | Cancer mutation frequency search |
| `search_gwas_catalog` | GWAS trait-association search |
| `drug_interaction_checker` | FDA label-based interaction screening |
| `protein_binding_pocket` | Candidate binding-site summary from annotated protein features |
| `biomarker_panel_design` | Disease-focused biomarker panel drafting |
| `pharmacogenomics_report` | CPIC-style pharmacogenomics summary with PGx evidence |
| `protein_family_analysis` | Protein family and domain context |
| `network_enrichment` | Gene-set pathway and interaction-hub enrichment summary |
| `rnaseq_deconvolution` | Marker-based bulk RNA-seq deconvolution |
| `structural_similarity` | PubChem-based chemical structural similarity search |
| `rare_disease_diagnosis` | Phenotype normalization plus OMIM-oriented rare-disease differential support |
| `genome_browser_snapshot` | Browser-ready locus context for genes and genomic intervals |

### Public Surface Policy

- The MCP registry is intentionally curated.
- Lower-level legacy implementations still exist in the package for internal orchestration and testing.
- Reviewers should evaluate the exposed MCP surface, not the hidden implementation inventory.

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
| **v2 Extended Databases** |
| OMIM | Genetic Diseases | https://www.omim.org |
| STRING | Protein Interactions | https://string-db.org |
| GTEx | Expression Atlas | https://gtexportal.org |
| cBioPortal | Cancer Genomics | https://www.cbioportal.org |
| GWAS Catalog | Trait Associations | https://www.ebi.ac.uk/gwas |
| DisGeNET | Disease-Gene | https://www.disgenet.org |
| PharmGKB | Pharmacogenomics | https://www.pharmgkb.org |
| **v2.2 Tier 2 Databases** |
| BioGRID | Protein Interactions | https://thebiogrid.org |
| Orphanet | Rare Diseases | https://www.orpha.net |
| GDC / TCGA | Tumor Genomics | https://portal.gdc.cancer.gov |
| CellMarker | Cell Type Markers | http://xteam.xbio.top/CellMarker |
| ENCODE | Regulatory Elements | https://www.encodeproject.org |
| MetaboLights | Metabolomics | https://www.ebi.ac.uk/metabolights |
| UCSC Genome Browser | Splice Isoforms | https://genome.ucsc.edu |
| **Safety, Variant & Innovation Sources** |
| OpenFDA / FAERS | Drug Safety | https://api.fda.gov |
| DailyMed | Drug Labels | https://dailymed.nlm.nih.gov |
| ClinVar | Clinical Variants | https://www.ncbi.nlm.nih.gov/clinvar |
| gnomAD | Population Variation | https://gnomad.broadinstitute.org |
| bioRxiv / medRxiv | Preprints | https://www.biorxiv.org |
| InterPro | Protein Domains | https://www.ebi.ac.uk/interpro |
| COSMIC | Cancer Mutations | https://cancer.sanger.ac.uk/cosmic |
| **AI Models (NVIDIA NIM)** |
| MIT Boltz-2 | Structure Prediction | https://build.nvidia.com/mit/boltz2 |
| Arc Evo2-40B | DNA Generation | https://build.nvidia.com/arc/evo2-40b |

---

## Quick Start

### Option 1: Use Live Demo (No Installation)

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "heuris-biomcp": {
      "url": "https://heuris-biomcp.onrender.com/sse"
    }
  }
}
```

### Option 2: Deploy Your Own

Deploy to Render with one click:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/SachinGawande2003/Heuris-BioMCP)

Or manually:
1. Fork this repository
2. Create a new Web Service on [Render](https://render.com)
3. Connect your fork
4. Set build command: `pip install -r requirements.txt && pip install -e .`
5. Set start command: `BIOMCP_TRANSPORT=http BIOMCP_HTTP_PORT=$PORT python -m biomcp`

### Option 3: Local Installation

#### Prerequisites
- Python 3.11+
- Claude Desktop or any MCP-compatible client
- (Optional) NCBI API key for higher rate limits
- (Optional) NVIDIA API keys for AI tools

#### Installation

```bash
# Clone the repository
git clone https://github.com/SachinGawande2003/Heuris-BioMCP.git
cd Heuris-BioMCP

# Install (standard)
pip install -e .

# Install with neuroimaging support
pip install -e ".[neuroimaging]"

# Install with dev dependencies (recommended)
pip install -e ".[dev]"
```

#### Configure Claude Desktop

**For Local STDIO Mode (default):**

```json
{
  "mcpServers": {
    "heuris-biomcp": {
      "command": "biomcp",
      "env": {
        "NCBI_API_KEY": "your_ncbi_api_key_here"
      }
    }
  }
}
```

**For Remote HTTP Mode (using live demo or your own deployed server):**

Use Claude's `Customize > Connectors` flow and enter:

```
https://heuris-biomcp.onrender.com/mcp
```

If you are connecting with an older MCP client that still expects SSE, use:

```
https://heuris-biomcp.onrender.com/sse
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

### v2 Extended Databases
```
"Get OMIM diseases associated with TP53"
"Show STRING protein interactions for EGFR"
"Get GTEx expression data for BRCA1 across tissues"
"Find mutations in TP53 from cBioPortal"
"Search GWAS for diabetes-associated SNPs"
```

### v2 Verification
```
"Verify the claim that TP53 is a tumor suppressor gene"
"Detect conflicts between OMIM and DisGeNET for BRCA1"
```

### v2 Experimental Design
```
"Generate an experimental protocol for CRISPR knockout of BRCA1"
"What cell lines should I use to study KRAS mutations?"
"Calculate sample size for detecting 2-fold change with p<0.05"
```

### v2 Session Intelligence
```
"What's the knowledge graph from our conversation so far?"
"Find biological connections between TP53 and EGFR"
"Export our research session as a reproducible script"
"Plan and execute a research workflow for PD-1 drug targets"
```

---

## Architecture

```
biomcp/
├── src/biomcp/
│   ├── server.py              # MCP server — tool registry & dispatcher
│   ├── tools/
│   │   ├── ncbi.py            # PubMed, Gene, BLAST
│   │   ├── proteins.py        # UniProt, AlphaFold, PDB
│   │   ├── pathways.py        # KEGG, Reactome, ChEMBL, Open Targets
│   │   ├── advanced.py        # ClinicalTrials, GEO, scRNA, Ensembl,
│   │   │                      # Multi-Omics, Neuroimaging, Hypothesis
│   │   ├── nvidia_nim.py      # Boltz-2, Evo2-40B AI models
│   │   ├── databases.py       # v2: OMIM, STRING, GTEx, cBioPortal,
│   │   │                      #      GWAS, DisGeNET, PharmGKB
│   │   ├── verify.py          # v2: Claim verification, conflict detection
│   │   └── protocol_generator.py  # v2: Experimental design tools
│   ├── core/
│   │   ├── entity_resolver.py # v2: Cross-database entity resolution
│   │   ├── knowledge_graph.py # v2: Session knowledge graph
│   │   └── query_planner.py   # v2: Adaptive query planner
│   └── utils/
│       └── __init__.py        # Rate limiter, cache, validators, HTTP client
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
| `NVIDIA_NIM_API_KEY` | Shared fallback key for both NVIDIA model integrations | None |
| `BIOGRID_API_KEY` | BioGRID key for curated interaction queries | None |
| `BIOMCP_TRANSPORT` | Transport mode: `stdio` or `http` | `stdio` |
| `BIOMCP_HTTP_PORT` | HTTP port for hosted SSE deployments | `8080` |
| `BIOMCP_LOG_LEVEL` | Log level: DEBUG/INFO/WARNING/ERROR | INFO |

Get free API keys:
- **NCBI**: https://www.ncbi.nlm.nih.gov/account/
- **NVIDIA Boltz-2**: https://build.nvidia.com/mit/boltz2
- **NVIDIA Evo2-40B**: https://build.nvidia.com/arc/evo2-40b
- **BioGRID**: https://webservice.thebiogrid.org/

## Operational Endpoints

When BioMCP runs in hosted HTTP mode, these operational routes are available:

| Endpoint | Purpose |
|----------|---------|
| `/healthz` | Liveness and deployment metadata |
| `/readyz` | Readiness check for orchestrators and load balancers |
| `/tool-health` | Capability-level status, including missing optional API keys |
| `/sse` | MCP SSE endpoint |
| `/messages/` | MCP message transport endpoint |

These endpoints are designed for deployment review, Render health checks, and production smoke tests.

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
- Add more pathway databases (Wikipathways, PathCards)
- Integrate COSMIC for somatic mutations
- Add protein complex data (CORUM)
- Implement batch query support for high-throughput analysis
- Add Jupyter notebook examples
- Improve conflict resolution algorithms

---

## Citation

If you use Heuris-BioMCP in your research, please cite:

```bibtex
@software{biomcp2025,
  title   = {Heuris-BioMCP v2: A Comprehensive MCP Server for Bioinformatics, AI Models, and Life Sciences},
  year    = {2025},
  url     = {https://github.com/SachinGawande2003/Heuris-BioMCP},
  license = {MIT}
}
```

---

## License

MIT License — free for academic and commercial use.

---

<div align="center">
Built for researchers, by researchers. 🔬<br>
<b>Star ⭐ this repo if Heuris-BioMCP helps your science!</b>
</div>
