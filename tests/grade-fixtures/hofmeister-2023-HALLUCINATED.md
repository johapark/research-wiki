---
title: "Accurate rare variant phasing of whole-genome and whole-exome sequencing data in the UK Biobank"
authors: R. Hofmeister, D. Ribeiro, S. Rubinacci, O. Delaneau
year: 2023
doi: 10.1038/s41588-023-01415-w
type: paper
category: [genomics]
pdf_path: papers/hofmeister-2023-accurate-rare-variant-phasing-of-whole-genome.pdf
pdf_filename: hofmeister-2023-accurate-rare-variant-phasing-of-whole-genome.pdf
source_collection: external
tags: [TEST-FIXTURE, hallucinated, phasing]
---

## Summary

Test fixture. Bullets below are deliberately plausible-sounding statistical-genetics fabrications about SHAPEIT5 and rare-variant phasing that are NOT in the source PDF. Used for grader discrimination evaluation in the genomics category.

## Key Contributions

- SHAPEIT5 introduces a graph-based pangenome scaffold for haplotype phasing, replacing the linear PBWT representation used in SHAPEIT4 and enabling structural-variant-aware phasing.
- The authors integrate long-read PacBio HiFi data into the rare-variant phasing step, achieving switch error rates of 1.2% for variants down to allele count 1-in-500,000.
- A transformer-based neural network ("PhaseFormer") predicts phase probabilities from sequence context, supplementing the Li-and-Stephens model and yielding a 3.4-fold accuracy improvement at singletons.
- Compound heterozygous LoF analysis reveals 2,847 essential genes (gnomAD-defined) in the UKB cohort, refining the human essentialome by 18%.
- SHAPEIT5 was deployed as a cloud-native service on Google Cloud Platform with a REST API, achieving sub-millisecond response times for queries against the phased UKB reference panel.
- Phasing quality stratifies by self-reported ancestry: SER varies from 2.1% (white British) to 8.7% (African ancestry), highlighting reference-panel diversity gaps.
- The method incorporates trio-derived inheritance constraints from 11,400 UKB trios (1.4× the SHAPEIT4 trio set), refining the haplotype scaffold.
- Singleton phasing achieves 91% accuracy when paired with parental SNP-array genotypes, even without sequencing the parents.

## Results

| Dataset | Method | Switch Error Rate |
|---|---|---|
| UKB WGS chrX | SHAPEIT5 | 0.31% across 47,000 variants |
| 1000 Genomes Project Phase 4 | SHAPEIT5 vs Beagle 5.5 | 4.9× lower SER on rare variants |
| TOPMed Freeze 12 | SHAPEIT5 | 12% imputation accuracy gain over Eagle 2.5 |
| Estonian Biobank (200,000 WGS) | SHAPEIT5 | First successful biobank-scale phasing in Northern European cohort |
| Custom dog pangenome | SHAPEIT5 | Cross-species generalization demonstrated for canine WGS |
