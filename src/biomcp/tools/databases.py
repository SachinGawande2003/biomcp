"""
BioMCP — Extended Database Tools
==================================
Seven major databases missing from the original BioMCP, now integrated:

  get_omim_gene_diseases     — OMIM: genetic disease ↔ gene relationships
  get_string_interactions    — STRING: protein-protein interaction networks
  get_gtex_expression        — GTEx: tissue-specific expression in healthy humans
  search_cbio_mutations      — cBioPortal: cancer mutation frequencies (TCGA)
  search_gwas_catalog        — GWAS Catalog: genome-wide association studies
  get_disgenet_associations  — DisGeNET: comprehensive gene-disease evidence
  get_pharmgkb_variants      — PharmGKB: pharmacogenomics (drug response by genotype)

APIs:
  OMIM            https://api.omim.org/api/ (requires key)
  STRING          https://string-db.org/api/
  GTEx Portal     https://gtexportal.org/api/v2/
  cBioPortal      https://www.cbioportal.org/api/
  GWAS Catalog    https://www.ebi.ac.uk/gwas/rest/api/
  DisGeNET        https://www.disgenet.org/api/
  PharmGKB        https://api.pharmgkb.org/v1/
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from biomcp.utils import (
    BioValidator,
    cached,
    get_http_client,
    rate_limited,
    with_retry,
)

STRING_BASE   = "https://string-db.org/api"
GTEX_BASE     = "https://gtexportal.org/api/v2"
CBIO_BASE     = "https://www.cbioportal.org/api"
GWAS_BASE     = "https://www.ebi.ac.uk/gwas/rest/api"
DISGENET_BASE = "https://www.disgenet.org/api"
PHARMGKB_BASE = "https://api.pharmgkb.org/v1"


# ─────────────────────────────────────────────────────────────────────────────
# OMIM — genetic disease database
# ─────────────────────────────────────────────────────────────────────────────

@cached("omim")
@rate_limited("default")
@with_retry(max_attempts=3)
async def get_omim_gene_diseases(
    gene_symbol: str,
) -> dict[str, Any]:
    """
    Retrieve OMIM disease entries associated with a gene.

    OMIM (Online Mendelian Inheritance in Man) is the authoritative
    database for genetic diseases. Uses the free Morbid Map endpoint.

    Args:
        gene_symbol: HGNC gene symbol (e.g. 'BRCA1', 'TP53', 'CFTR').

    Returns:
        {
          gene, omim_gene_entry, diseases: [{ omim_id, phenotype,
          inheritance_pattern, gene_mim, phenotype_mim, mapping_key }]
        }

    Inheritance patterns: AD=Autosomal dominant, AR=Autosomal recessive,
        XL=X-linked, XLD=X-linked dominant, XLR=X-linked recessive,
        MT=Mitochondrial
    """
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    client      = await get_http_client()

    # Use NCBI Gene → OMIM cross-reference as primary pathway
    # (OMIM API requires commercial key; we use the public NCBI link-out)
    from biomcp.utils import ncbi_params

    # Step 1: Get NCBI Gene ID
    esearch = await client.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params=ncbi_params({
            "db":   "gene",
            "term": f"{gene_symbol}[Gene Name] AND Homo sapiens[Organism]",
            "retmax": 1,
        }),
    )
    esearch.raise_for_status()
    ids = esearch.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return {"gene": gene_symbol, "diseases": [], "error": f"'{gene_symbol}' not found in NCBI Gene."}

    gene_id = ids[0]

    # Step 2: Get OMIM links via elink
    elink = await client.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi",
        params=ncbi_params({
            "dbfrom": "gene",
            "db":     "omim",
            "id":     gene_id,
        }),
    )
    elink.raise_for_status()
    elink_text = elink.text

    # Parse OMIM IDs from XML
    import xml.etree.ElementTree as ET
    omim_ids: list[str] = []
    try:
        root = ET.fromstring(elink_text)
        for link in root.findall(".//Link"):
            oid = link.findtext("Id")
            if oid:
                omim_ids.append(oid)
    except ET.ParseError:
        pass

    if not omim_ids:
        return {"gene": gene_symbol, "ncbi_gene_id": gene_id, "diseases": [], "omim_ids": []}

    # Step 3: Fetch OMIM entry details via esummary
    omim_ids_capped = omim_ids[:10]
    esumm = await client.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params=ncbi_params({"db": "omim", "id": ",".join(omim_ids_capped)}),
    )
    esumm.raise_for_status()
    omim_data = esumm.json().get("result", {})

    diseases: list[dict[str, Any]] = []
    for omim_id in omim_ids_capped:
        entry = omim_data.get(omim_id, {})
        if not entry:
            continue
        title     = entry.get("title", "")
        omim_type = entry.get("omimtype", "")
        if not title:
            continue

        # Parse inheritance pattern from title
        inheritance = ""
        if ", autosomal dominant" in title.lower():
            inheritance = "Autosomal Dominant (AD)"
        elif ", autosomal recessive" in title.lower():
            inheritance = "Autosomal Recessive (AR)"
        elif ", x-linked" in title.lower():
            inheritance = "X-Linked (XL)"
        elif ", mitochondrial" in title.lower():
            inheritance = "Mitochondrial (MT)"

        diseases.append({
            "omim_id":           omim_id,
            "phenotype":         title,
            "omim_type":         omim_type,
            "inheritance_pattern": inheritance,
            "omim_url":          f"https://omim.org/entry/{omim_id}",
        })

    return {
        "gene":         gene_symbol,
        "ncbi_gene_id": gene_id,
        "total_omim_entries": len(omim_ids),
        "diseases":     diseases,
        "omim_gene_url": f"https://omim.org/search?search=gene:{gene_symbol}",
        "note": "OMIM data retrieved via NCBI elink. For full phenotype descriptions, visit OMIM directly.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRING — protein-protein interaction network
# ─────────────────────────────────────────────────────────────────────────────

@cached("string")
@rate_limited("default")
@with_retry(max_attempts=3)
async def get_string_interactions(
    gene_symbol: str,
    min_score:   int = 400,
    max_results: int = 20,
    species:     int = 9606,
) -> dict[str, Any]:
    """
    Retrieve protein-protein interaction network from STRING database.

    STRING integrates evidence from: co-expression, gene fusion, co-occurrence,
    experiments, databases, text mining, and neighborhood.

    Args:
        gene_symbol: HGNC gene symbol.
        min_score:   Minimum interaction confidence score (0–1000).
                     400=medium, 700=high, 900=very high. Default 400.
        max_results: Interaction partners to return (1–100). Default 20.
        species:     NCBI taxonomy ID. Default 9606 (Homo sapiens).

    Returns:
        {
          gene, string_id, interactions: [{ partner, combined_score,
          experimental_score, coexpression_score, textmining_score,
          database_score, string_url }],
          network_url, stats
        }

    Score interpretation:
        > 900  Very high confidence
        700–900 High confidence
        400–700 Medium confidence
        150–400 Low confidence
    """
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    min_score   = BioValidator.clamp_int(min_score, 0, 1000, "min_score")
    max_results = BioValidator.clamp_int(max_results, 1, 100, "max_results")
    client      = await get_http_client()

    # Step 1: Resolve gene to STRING ID
    resolve_resp = await client.get(
        f"{STRING_BASE}/json/get_string_ids",
        params={
            "identifiers": gene_symbol,
            "species":     species,
            "limit":       1,
            "echo_query":  1,
            "caller_identity": "BioMCP",
        },
    )
    resolve_resp.raise_for_status()
    resolved = resolve_resp.json()
    if not resolved:
        return {
            "gene":   gene_symbol,
            "error":  f"'{gene_symbol}' not found in STRING database.",
            "interactions": [],
        }

    string_id = resolved[0].get("stringId", "")

    # Step 2: Get interaction network
    net_resp = await client.get(
        f"{STRING_BASE}/json/network",
        params={
            "identifiers":     string_id,
            "species":         species,
            "required_score":  min_score,
            "limit":           max_results,
            "caller_identity": "BioMCP",
        },
    )
    net_resp.raise_for_status()
    network = net_resp.json()

    interactions: list[dict[str, Any]] = []
    seen_partners: set[str] = set()

    for link in network:
        partner_a = link.get("preferredName_A", "")
        partner_b = link.get("preferredName_B", "")
        partner   = partner_b if partner_a.upper() == gene_symbol else partner_a
        if not partner or partner in seen_partners:
            continue
        seen_partners.add(partner)

        combined = link.get("score", 0)
        interactions.append({
            "partner":            partner,
            "combined_score":     combined,
            "experimental_score": link.get("escore", 0),
            "database_score":     link.get("dscore", 0),
            "coexpression_score": link.get("ascore", 0),
            "textmining_score":   link.get("tscore", 0),
            "cooccurrence_score": link.get("oscore", 0),
            "confidence": (
                "Very high" if combined >= 0.9 else
                "High"      if combined >= 0.7 else
                "Medium"    if combined >= 0.4 else
                "Low"
            ),
            "string_url": (
                f"https://string-db.org/network/{string_id}%0D"
                f"{link.get('stringId_B', '')}"
            ),
        })

    # Sort by combined score descending
    interactions.sort(key=lambda x: x["combined_score"], reverse=True)

    return {
        "gene":             gene_symbol,
        "string_id":        string_id,
        "min_score_filter": min_score,
        "total_interactions": len(interactions),
        "interactions":     interactions,
        "network_url":      f"https://string-db.org/cgi/network?identifiers={gene_symbol}&species={species}",
        "stats": {
            "very_high_confidence": sum(1 for i in interactions if i["combined_score"] >= 0.9),
            "high_confidence":      sum(1 for i in interactions if 0.7 <= i["combined_score"] < 0.9),
            "medium_confidence":    sum(1 for i in interactions if 0.4 <= i["combined_score"] < 0.7),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# GTEx — tissue-specific gene expression in healthy humans
# ─────────────────────────────────────────────────────────────────────────────

@cached("gtex")
@rate_limited("default")
@with_retry(max_attempts=3)
async def get_gtex_expression(
    gene_symbol:    str,
    top_tissues:    int = 10,
    dataset_id:     str = "gtex_v8",
) -> dict[str, Any]:
    """
    Retrieve tissue-specific gene expression from GTEx (Genotype-Tissue Expression).

    GTEx provides RNA-seq data from 54 human tissues from ~1000 donors.
    Critical for understanding where a gene is expressed in healthy tissue —
    complementing GEO disease-state datasets.

    Args:
        gene_symbol: HGNC gene symbol.
        top_tissues: Tissues to return sorted by median TPM. Default 10.
        dataset_id:  GTEx dataset version. Default 'gtex_v8' (latest stable).

    Returns:
        {
          gene, versioned_gene_id, unit,
          expression_by_tissue: [{ tissue_site, tissue_site_detail,
          median_tpm, mean_tpm, std_tpm, n_samples }],
          highest_expressing_tissues: [...],
          lowest_expressing_tissues: [...],
          ubiquitous: bool
        }
    """
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    top_tissues = BioValidator.clamp_int(top_tissues, 1, 54, "top_tissues")
    client      = await get_http_client()

    resp = await client.get(
        f"{GTEX_BASE}/expression/geneExpression",
        params={
            "gencodeId":  gene_symbol,
            "datasetId":  dataset_id,
        },
        headers={"Accept": "application/json"},
    )

    if resp.status_code == 404:
        # Try by gene symbol lookup
        gene_resp = await client.get(
            f"{GTEX_BASE}/reference/gene",
            params={"geneSymbol": gene_symbol, "gencodeVersion": "v26"},
        )
        if gene_resp.status_code != 200:
            return {
                "gene":  gene_symbol,
                "error": f"Gene '{gene_symbol}' not found in GTEx {dataset_id}.",
                "expression_by_tissue": [],
            }
        genes = gene_resp.json().get("data", {}).get("gene", [])
        if not genes:
            return {"gene": gene_symbol, "error": "Not found in GTEx.", "expression_by_tissue": []}

        gencode_id = genes[0].get("gencodeId", "")
        resp = await client.get(
            f"{GTEX_BASE}/expression/geneExpression",
            params={"gencodeId": gencode_id, "datasetId": dataset_id},
        )

    resp.raise_for_status()
    data = resp.json().get("data", {})

    expression: list[dict[str, Any]] = []
    for entry in data.get("geneExpression", []):
        tissue_name = entry.get("tissueSiteDetailId", entry.get("tissueSite", ""))
        data_list   = entry.get("data", [])
        if not data_list:
            continue

        import statistics
        median_tpm = entry.get("median", statistics.median(data_list) if data_list else 0)
        mean_tpm   = entry.get("mean", sum(data_list) / len(data_list) if data_list else 0)

        expression.append({
            "tissue_site":        entry.get("tissueSite", ""),
            "tissue_site_detail": tissue_name,
            "median_tpm":         round(float(median_tpm), 3),
            "mean_tpm":           round(float(mean_tpm), 3),
            "n_samples":          len(data_list),
            "gtex_url":           f"https://gtexportal.org/home/gene/{gene_symbol}",
        })

    # Sort by median TPM
    expression.sort(key=lambda x: x["median_tpm"], reverse=True)

    all_medians  = [e["median_tpm"] for e in expression]
    ubiquitous   = len([m for m in all_medians if m > 1.0]) > 0.7 * len(all_medians) if all_medians else False

    return {
        "gene":             gene_symbol,
        "dataset":          dataset_id,
        "unit":             "TPM (Transcripts Per Million)",
        "total_tissues":    len(expression),
        "expression_by_tissue": expression[:top_tissues],
        "highest_expressing_tissues": [e["tissue_site_detail"] for e in expression[:5]],
        "lowest_expressing_tissues":  [e["tissue_site_detail"] for e in expression[-5:] if expression],
        "ubiquitously_expressed": ubiquitous,
        "max_median_tpm":   max(all_medians) if all_medians else 0,
        "gtex_gene_url":    f"https://gtexportal.org/home/gene/{gene_symbol}",
        "interpretation": (
            "Broadly expressed across most tissues (housekeeping gene)."
            if ubiquitous else
            f"Tissue-enriched — highest in: {', '.join([e['tissue_site_detail'] for e in expression[:3]])}"
            if expression else
            "No expression data available."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# cBioPortal — cancer mutation frequencies (TCGA + others)
# ─────────────────────────────────────────────────────────────────────────────

@cached("cbio")
@rate_limited("default")
@with_retry(max_attempts=3)
async def search_cbio_mutations(
    gene_symbol:  str,
    cancer_type:  str = "",
    max_studies:  int = 10,
) -> dict[str, Any]:
    """
    Search cBioPortal for cancer mutation frequencies across TCGA cohorts.

    cBioPortal hosts comprehensive cancer genomics data including somatic
    mutations, copy number alterations, and expression data across thousands
    of tumor samples.

    Args:
        gene_symbol: HGNC gene symbol.
        cancer_type: Cancer type filter (e.g. 'luad', 'brca', 'coad').
                     Leave empty for pan-cancer.
        max_studies: Studies to return (1–50). Default 10.

    Returns:
        {
          gene, cancer_type, studies: [{ study_id, name, cancer,
          n_samples, mutation_frequency_pct, altered_samples,
          top_mutations: [...] }],
          pan_cancer_summary
        }
    """
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    max_studies = BioValidator.clamp_int(max_studies, 1, 50, "max_studies")
    client      = await get_http_client()

    # Get available studies
    studies_resp = await client.get(
        f"{CBIO_BASE}/studies",
        params={"projection": "SUMMARY", "pageSize": 50, "pageNumber": 0},
        headers={"Accept": "application/json"},
    )
    studies_resp.raise_for_status()
    all_studies = studies_resp.json()

    # Filter by cancer type if specified
    if cancer_type:
        filtered = [s for s in all_studies if cancer_type.lower() in s.get("cancerTypeId", "").lower()]
    else:
        # Use major TCGA studies
        tcga_ids = {
            "tcga_luad", "tcga_brca", "tcga_coad", "tcga_prad",
            "tcga_gbm", "tcga_ov", "tcga_blca", "tcga_kirc",
            "tcga_stad", "tcga_skcm", "tcga_hnsc", "tcga_lihc",
        }
        filtered = [s for s in all_studies if s.get("studyId", "").lower() in tcga_ids]

    filtered = filtered[:max_studies]

    # Query mutation data for each study in parallel
    import asyncio as _asyncio

    async def _query_study(study: dict) -> dict[str, Any] | None:
        study_id = study.get("studyId", "")
        try:
            mut_resp = await client.get(
                f"{CBIO_BASE}/studies/{study_id}/genes/{gene_symbol}/mutations",
                headers={"Accept": "application/json"},
                params={"projection": "SUMMARY"},
            )
            if mut_resp.status_code not in (200, 206):
                return None
            mutations = mut_resp.json()
            if not mutations:
                return None

            # Count samples and get top mutation types
            sample_ids: set[str] = set()
            mutation_types: dict[str, int] = {}
            for mut in mutations:
                sid = mut.get("sampleId", "")
                if sid:
                    sample_ids.add(sid)
                mut_type = mut.get("mutationType", "") or mut.get("variantClassification", "")
                if mut_type:
                    mutation_types[mut_type] = mutation_types.get(mut_type, 0) + 1

            n_samples = study.get("allSampleCount", 1)
            n_altered = len(sample_ids)
            freq_pct  = round(n_altered / max(n_samples, 1) * 100, 2)

            top_muts = sorted(mutation_types.items(), key=lambda x: x[1], reverse=True)[:5]

            return {
                "study_id":               study_id,
                "study_name":             study.get("name", ""),
                "cancer_type":            study.get("cancerTypeId", ""),
                "total_samples":          n_samples,
                "altered_samples":        n_altered,
                "mutation_frequency_pct": freq_pct,
                "top_mutation_types":     [{"type": t, "count": c} for t, c in top_muts],
                "cbio_url":               f"https://www.cbioportal.org/results/mutations?gene={gene_symbol}&study={study_id}",
            }
        except Exception as exc:
            logger.debug(f"[cBioPortal] Study {study_id} failed: {exc}")
            return None

    study_results_raw = await _asyncio.gather(*[_query_study(s) for s in filtered])
    studies_with_data = [r for r in study_results_raw if r is not None]
    studies_with_data.sort(key=lambda x: x["mutation_frequency_pct"], reverse=True)

    # Pan-cancer summary
    total_altered = sum(s["altered_samples"]    for s in studies_with_data)
    total_samples = sum(s["total_samples"]       for s in studies_with_data)
    pan_cancer_freq = round(total_altered / max(total_samples, 1) * 100, 2) if total_samples else 0

    return {
        "gene":                gene_symbol,
        "cancer_type_filter":  cancer_type or "pan-cancer",
        "studies_queried":     len(filtered),
        "studies_with_mutations": len(studies_with_data),
        "studies":             studies_with_data,
        "pan_cancer_summary": {
            "total_samples_analyzed":    total_samples,
            "total_altered_samples":     total_altered,
            "pan_cancer_frequency_pct":  pan_cancer_freq,
            "highest_frequency_cancer":  studies_with_data[0]["cancer_type"] if studies_with_data else "",
            "highest_frequency_pct":     studies_with_data[0]["mutation_frequency_pct"] if studies_with_data else 0,
        },
        "cbio_gene_url": f"https://www.cbioportal.org/results/mutations?gene={gene_symbol}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# GWAS Catalog — genome-wide association study hits
# ─────────────────────────────────────────────────────────────────────────────

@cached("gwas")
@rate_limited("default")
@with_retry(max_attempts=3)
async def search_gwas_catalog(
    gene_symbol: str,
    p_value_threshold: float = 5e-8,
    max_results: int = 20,
) -> dict[str, Any]:
    """
    Search the NHGRI-EBI GWAS Catalog for genome-wide significant associations.

    The GWAS Catalog contains all published GWAS, providing a comprehensive
    picture of the genetic architecture of complex traits.

    Args:
        gene_symbol:        HGNC gene symbol.
        p_value_threshold:  Max p-value for significance. Default 5e-8 (GWAS threshold).
        max_results:        Associations to return (1–100). Default 20.

    Returns:
        {
          gene, total_associations, associations: [{ snp_id, trait,
          p_value, odds_ratio, beta, risk_allele, study_pmid,
          mapped_genes, reported_genes, gwas_url }]
        }
    """
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    max_results = BioValidator.clamp_int(max_results, 1, 100, "max_results")
    client      = await get_http_client()

    resp = await client.get(
        f"{GWAS_BASE}/associations/search",
        params={
            "q":    f"mappedGenes:{gene_symbol}",
            "size": max_results,
            "sort": "pValueMantissa:asc",
        },
        headers={"Accept": "application/json"},
    )
    if resp.status_code == 404:
        return {"gene": gene_symbol, "total_associations": 0, "associations": []}
    resp.raise_for_status()

    content = resp.json()
    raw_assocs = (
        content.get("_embedded", {}).get("associations", [])
        or content.get("associations", [])
    )

    associations: list[dict[str, Any]] = []
    for assoc in raw_assocs:
        p_mant    = assoc.get("pvalueMantissa", 1)
        p_exp     = assoc.get("pvalueExponent", 0)
        p_value   = p_mant * (10 ** p_exp) if p_mant and p_exp else None

        if p_value and p_value > p_value_threshold:
            continue

        # Extract SNP info
        loci      = assoc.get("loci", [])
        snp_ids: list[str] = []
        risk_alleles: list[str] = []
        for locus in loci:
            for snp in locus.get("strongestRiskAlleles", []):
                risk_allele = snp.get("riskAlleleName", "")
                if "-" in risk_allele:
                    parts = risk_allele.split("-")
                    snp_ids.append(parts[0])
                    risk_alleles.append(risk_allele)

        # Extract trait from embedded study
        trait    = assoc.get("traitName", "")
        study    = assoc.get("study", {})
        if isinstance(study, dict):
            trait = study.get("diseaseTrait", {}).get("trait", trait)

        pmid     = ""
        if isinstance(study, dict):
            for pub in study.get("publications", []):
                pmid = pub.get("pubmedId", "")
                break

        associations.append({
            "snp_ids":       snp_ids,
            "risk_alleles":  risk_alleles,
            "trait":         trait,
            "p_value":       p_value,
            "p_value_str":   f"{p_mant}e{p_exp}" if p_mant else "",
            "odds_ratio":    assoc.get("orPerCopyNum"),
            "beta":          assoc.get("betaNum"),
            "ci_text":       assoc.get("range", ""),
            "study_pmid":    pmid,
            "pubmed_url":    f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            "gwas_url":      f"https://www.ebi.ac.uk/gwas/associations/{assoc.get('accessionId', '')}",
        })

    return {
        "gene":                gene_symbol,
        "p_value_threshold":   p_value_threshold,
        "total_associations":  content.get("page", {}).get("totalElements", len(associations)),
        "returned":            len(associations),
        "associations":        associations,
        "gwas_gene_url":       f"https://www.ebi.ac.uk/gwas/genes/{gene_symbol}",
        "note": (
            "GWAS hits mapped to gene region — may not imply direct causality. "
            "Fine-mapping required to identify causal variant."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DisGeNET — comprehensive gene-disease evidence
# ─────────────────────────────────────────────────────────────────────────────

@cached("disgenet")
@rate_limited("default")
@with_retry(max_attempts=3)
async def get_disgenet_associations(
    gene_symbol: str,
    min_score:   float = 0.1,
    max_results: int = 20,
) -> dict[str, Any]:
    """
    Retrieve gene-disease associations from DisGeNET.

    DisGeNET integrates evidence from expert-curated databases (UniProt,
    OMIM, Orphanet, CGI), GWAS, animal models, and literature mining —
    providing a more comprehensive disease atlas than Open Targets alone.

    Args:
        gene_symbol: HGNC gene symbol.
        min_score:   Minimum GDA (Gene-Disease Association) score (0–1).
                     DisGeNET score weighs source reliability + evidence count.
        max_results: Associations (1–100). Default 20.

    Returns:
        {
          gene, total_associations, associations: [{ disease_name,
          disease_type, gda_score, evidence_index, disease_classes,
          source_count, pmid_count, disgenet_url }]
        }
    """
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    max_results = BioValidator.clamp_int(max_results, 1, 100, "max_results")
    client      = await get_http_client()

    resp = await client.get(
        f"{DISGENET_BASE}/gda/gene/{gene_symbol}",
        params={
            "min_score":     min_score,
            "limit":         max_results,
            "format":        "json",
        },
        headers={
            "Accept":      "application/json",
            "User-Agent":  "BioMCP/1.0 (research; +https://github.com/SachinGawande2003/biomcp)",
        },
    )

    if resp.status_code == 404:
        return {"gene": gene_symbol, "total_associations": 0, "associations": []}
    if resp.status_code in (401, 403):
        return {
            "gene":  gene_symbol,
            "error": (
                "DisGeNET requires authentication for high-volume queries. "
                "Visit https://www.disgenet.org/api/ for API key registration. "
                "Using Open Targets as an alternative: use get_gene_disease_associations."
            ),
            "associations": [],
        }
    resp.raise_for_status()

    data = resp.json()
    raw  = data if isinstance(data, list) else data.get("data", [])

    associations: list[dict[str, Any]] = []
    for entry in raw[:max_results]:
        disease_name = entry.get("diseaseName", "")
        if not disease_name:
            continue
        associations.append({
            "disease_id":      entry.get("diseaseId", ""),
            "disease_name":    disease_name,
            "disease_type":    entry.get("diseaseType", ""),
            "disease_classes": entry.get("diseaseClasses", []),
            "gda_score":       round(float(entry.get("score", 0)), 4),
            "ei":              round(float(entry.get("EI", 0)), 4),   # Evidence Index
            "dsi":             round(float(entry.get("DSI", 0)), 4),  # Disease Specificity Index
            "dpi":             round(float(entry.get("DPI", 0)), 4),  # Disease Pleiotropy Index
            "pmid_count":      entry.get("pmidCount", 0),
            "source_count":    entry.get("sourceCount", 0),
            "disgenet_url":    f"https://www.disgenet.org/browser/1/1/0/1/geneid__{entry.get('geneId', '')}--diseaseId__{entry.get('diseaseId', '')}/",
        })

    associations.sort(key=lambda x: x["gda_score"], reverse=True)

    return {
        "gene":               gene_symbol,
        "min_score_filter":   min_score,
        "total_associations": len(associations),
        "associations":       associations,
        "disgenet_gene_url":  f"https://www.disgenet.org/search/#summary/{gene_symbol}/gene",
        "score_interpretation": {
            "high_0.6_1.0":   "Extensively curated from multiple high-quality sources",
            "medium_0.3_0.6": "Supported by automated text mining + some curation",
            "low_0.1_0.3":    "Primarily text mining evidence — verify manually",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# PharmGKB — pharmacogenomics (drug response by genotype)
# ─────────────────────────────────────────────────────────────────────────────

@cached("pharmgkb")
@rate_limited("default")
@with_retry(max_attempts=3)
async def get_pharmgkb_variants(
    gene_symbol: str,
    max_results: int = 15,
) -> dict[str, Any]:
    """
    Retrieve pharmacogenomics data from PharmGKB.

    PharmGKB curates how genetic variation affects drug response —
    critical for personalized medicine and understanding why patients
    respond differently to the same drug.

    Args:
        gene_symbol: HGNC gene symbol (e.g. 'CYP2D6', 'TPMT', 'DPYD').
        max_results: Variant-drug annotations to return. Default 15.

    Returns:
        {
          gene, pharmgkb_gene_id,
          clinical_annotations: [{ variant, drug, phenotype_category,
          significance, evidence_level, pmids }],
          variant_drug_annotations: [...],
          pathways: [...]
        }

    Evidence levels (Clinical Annotation):
        1A — FDA label + expert consensus (highest)
        1B — FDA label
        2A — Meta-analysis / well-powered study
        2B — Moderate evidence
        3  — Limited evidence
        4  — Case report only (lowest)
    """
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    max_results = BioValidator.clamp_int(max_results, 1, 50, "max_results")
    client      = await get_http_client()

    # Step 1: Find gene in PharmGKB
    gene_resp = await client.get(
        f"{PHARMGKB_BASE}/gene",
        params={"symbol": gene_symbol, "view": "base"},
        headers={"Accept": "application/json"},
    )

    if gene_resp.status_code == 404:
        return {
            "gene":  gene_symbol,
            "error": f"'{gene_symbol}' not found in PharmGKB.",
            "clinical_annotations": [],
        }

    if gene_resp.status_code in (401, 403, 429):
        return {
            "gene":  gene_symbol,
            "note":  "PharmGKB API rate limited. Visit https://www.pharmgkb.org/ directly.",
            "clinical_annotations": [],
            "pharmgkb_url": f"https://www.pharmgkb.org/gene?symbol={gene_symbol}",
        }

    gene_resp.raise_for_status()
    gene_data   = gene_resp.json()
    gene_result = gene_data.get("data") or (gene_data if isinstance(gene_data, dict) else {})
    if isinstance(gene_result, list):
        gene_result = gene_result[0] if gene_result else {}

    pharmgkb_id = gene_result.get("id", "")

    # Step 2: Get clinical annotations for this gene
    clinical_annotations: list[dict[str, Any]] = []
    if pharmgkb_id:
        ca_resp = await client.get(
            f"{PHARMGKB_BASE}/clinicalAnnotation",
            params={
                "gene":  pharmgkb_id,
                "view":  "base",
                "size":  max_results,
            },
            headers={"Accept": "application/json"},
        )
        if ca_resp.status_code == 200:
            ca_data = ca_resp.json()
            raw_ca  = ca_data.get("data", ca_data if isinstance(ca_data, list) else [])
            for ca in raw_ca[:max_results]:
                variants = [v.get("name", "") for v in ca.get("variants", [])]
                drugs    = [d.get("name", "") for d in ca.get("chemicals", [])]
                clinical_annotations.append({
                    "variants":           variants,
                    "drugs":              drugs,
                    "phenotype_category": ca.get("phenotypeCategory", ""),
                    "significance":       ca.get("significance", ""),
                    "evidence_level":     ca.get("evidenceLevel", ""),
                    "evidence_level_description": _pharmgkb_level_description(
                        ca.get("evidenceLevel", "")
                    ),
                    "pmid_count":        len(ca.get("literature", [])),
                    "pharmgkb_url":      f"https://www.pharmgkb.org/clinicalAnnotation/{ca.get('id', '')}",
                })

    return {
        "gene":                gene_symbol,
        "pharmgkb_gene_id":    pharmgkb_id,
        "total_clinical_annotations": len(clinical_annotations),
        "clinical_annotations":       clinical_annotations,
        "pharmgkb_gene_url":   f"https://www.pharmgkb.org/gene?symbol={gene_symbol}",
        "clinical_relevance": (
            f"{gene_symbol} has {len(clinical_annotations)} pharmacogenomic annotations — "
            "clinically important for drug dosing."
            if len(clinical_annotations) >= 3 else
            f"Limited pharmacogenomic data for {gene_symbol} in PharmGKB."
        ),
        "important_note": (
            "PharmGKB Level 1A/1B annotations are embedded in FDA drug labels. "
            "Always consult current prescribing information for clinical decisions."
        ),
    }


def _pharmgkb_level_description(level: str) -> str:
    desc = {
        "1A": "FDA label + CPIC/DPWG guidelines — highest clinical evidence",
        "1B": "FDA label variant annotation",
        "2A": "Annotation + replicated or well-powered study",
        "2B": "Annotation + moderate evidence study",
        "3":  "Annotation + limited evidence",
        "4":  "Case report / very limited evidence",
    }
    return desc.get(str(level), f"Evidence level {level}")
