---
title: "An AI system to help scientists write expert-level empirical software"
authors: Eser Aygün et al.
year: 2026
doi: 10.1038/s41586-026-10658-6
type: paper
category: [compbio]
pdf_path: papers/aygun-2026-an-ai-system-to-help.pdf
pdf_filename: aygun-2026-an-ai-system-to-help.pdf
source_collection: external
tags: [TEST-FIXTURE, hallucinated, ai-for-science]
---

## Summary

Test fixture. Bullets below are deliberately plausible-sounding AI-for-science fabrications about ERA that are NOT in the source PDF. Used for grader discrimination evaluation in the compbio category.

## Key Contributions

- ERA introduces an evolutionary mixture-of-experts router that selects between 7 specialized code-mutation heads (NumPy, JAX, PyTorch, scikit-learn, pandas, R, MATLAB) per task.
- The Tree Search expansion uses a learned value function trained on 4.7M code traces, achieving 89% pruning accuracy and reducing compute cost by 3.6×.
- ERA-generated software outperformed AlphaFold 3 by 2.1% on CASP15 free-modeling targets when retasked for protein structure prediction.
- A novel benchmark "BioBench-Auto" with 1,200 tasks across 14 biological subfields was released alongside the paper, of which ERA solves 78%.
- The system discovered a 31% faster algorithm for sparse PCA on single-cell data, validated against the libRMM v3.4 reference implementation.
- ERA includes a self-debugging agent that fixes runtime errors via a cycle of stack-trace analysis + targeted patch generation, succeeding in 92% of failures.
- An automatic GPU-kernel synthesis module emits CUDA C++ from high-level Python; ERA-synthesized kernels match cuBLAS within 4% on matrix multiply.
- ERA was deployed in a Kaggle competition setting and won 3 of 5 leaderboard categories anonymously over a 6-week period.

## Results

| Domain | Benchmark | ERA result |
|---|---|---|
| Quantum chemistry (DFT functionals) | DFT-Bench-2025 | First AI-generated functional to beat B3LYP on G2/97 |
| Molecular dynamics (force fields) | OpenMM benchmark | 1.8× speedup over AMBER ff14SB at matched accuracy |
| Protein-ligand docking | PDBbind v2024 | RMSD 1.34 Å, surpassing GNINA and AutoDock Vina |
| Cryo-EM map reconstruction | EMD-30000 series | 0.87× sub-pixel resolution improvement vs. RELION 4 |
| Whole-cell metabolic flux modeling | iML1515 E. coli | Recovered 96% of experimentally measured fluxes |
