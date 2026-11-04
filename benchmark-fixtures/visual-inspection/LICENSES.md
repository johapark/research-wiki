# Visual benchmark licenses and attribution

The dataset files in this directory (`assets/*.png`, `cases.json`,
`nomination-cases.json`, `manifest.json`, and `reference/*.json`) are available
under [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/).
They may be used, shared and adapted, including commercially, with attribution,
a license link and an indication of changes. Preserve this file when
redistributing the dataset. The Python utilities retain the repository's
[PolyForm Noncommercial license](../../LICENSE); this dataset license does not
change the framework's license.

Original paper material retains its authors' copyright and CC-BY-4.0 terms.
Benchmark questions and reference annotations were prepared for research-wiki
on 2026-11-04. Credit research-wiki for those additions and the paper authors
below for the source images and excerpts. The PDFs are already bundled in
[`../pdfs/`](../pdfs/) with attribution in [`../LICENSES.md`](../LICENSES.md).

Licenses were checked against the exact bundled PDFs on 2026-11-04. Each
license grant is bound to its article by the title or DOI on the license page,
within that article's own boundary. The inspected tables, captions and figure plates carry no
separate third-party credit line excluding them from the article license.

## MGA

- **Title:** MGA: a tool for haplotype-mixed assembly of long and accurate reads
- **Authors:** Zhenmiao Zhang, Marcus W. Fedarko, Anton Bankevich, Pavel A. Pevzner
- **Source:** Genome Biology 27:201 (2026),
  [doi:10.1186/s13059-026-04128-5](https://doi.org/10.1186/s13059-026-04128-5)
- **Copyright/license:** © The Author(s) 2026; CC-BY-4.0.
- **License evidence:** Physical PDF page 1 explicitly grants Creative Commons
  Attribution 4.0 International reuse, including sharing and adaptation. The
  same page names the article and DOI. Its images are included unless separately
  credited otherwise.
- **PDF:** `../pdfs/zhang-2026-mga-a-tool-for-haplotype-mixed.pdf`
- **SHA-256:** `abdacdee6b5dc6b75fe398dc5d4557bd14d34b16eaddf4b04f71e3cd818fcdf0`
- **Material used:** Table 2 on physical page 10, Table 3 on page 17.
- **Changes:** Full pages rendered to RGB PNG at 110 DPI. The
  `mga-p10-masked@110.png` variant additionally applies two white masks to hide
  the target value and its repeated occurrence in a header; exact changes are
  recorded in [`reference/transformations.json`](reference/transformations.json).
  Nomination case n02 contains a caption-only text excerpt.

## Clinical trial

- **Title:** Adjunctive ibuprofen in pre-extensively drug-resistant and
  extensively drug-resistant tuberculosis: a phase IIA open-label pilot clinical trial
- **Authors:** Kaori L. Fonseca, Judith Farrés, Ketevan Barbakadze, Iza Jikia,
  Malkhaz Tsotskhalashvili, Juan M. García-Illarramendi, Lilibeth Arias,
  Kennedy Otwombe, Chiara Sopegno, Albert Despuig, Neil Martinson,
  Nadiia Buhiichyk, Tamta Korinteli, Émilie Layre, Jérôme Nigou, Zaza Avaliani,
  Nestani Tukvadze, Sergo Vashakidze, Cristina Vilaplana
- **Source:** Nature Communications (2026),
  [doi:10.1038/s41467-026-75148-9](https://doi.org/10.1038/s41467-026-75148-9),
  the bundled article-in-press version.
- **Copyright/license:** © The Author(s) 2026; CC-BY-4.0.
- **License evidence:** Physical PDF page 1 explicitly grants Creative Commons
  Attribution 4.0 International reuse, including sharing and adaptation. The
  same page names the article, authors and DOI. Its images are included unless
  separately credited otherwise.
- **PDF:** `../pdfs/fonseca-2026-adjunctive-ibuprofen-in-pre-extensively-drug-resistant.pdf`
- **SHA-256:** `7298a9e00b1097bf01805ef81964624e338a32b27dc491d151b9eb7fc78ebc8e`
- **Material used:** Figure captions on physical page 29; Figure 2 forest plate
  on page 35; Figure 3 box plot on page 36; Results excerpt on page 7.
- **Changes:** Full pages rendered to RGB PNG at 110 DPI without changing the
  scientific content. Nomination n01 uses an exact bounded Results excerpt;
  n03 uses caption titles only, deliberately withholding numerical legends.

## Tangermeme

- **Title:** Tangermeme: a toolkit for understanding cis-regulatory logic using deep learning models
- **Author:** Jacob Schreiber
- **Source:** Nature Methods (2026),
  [doi:10.1038/s41592-026-03254-z](https://doi.org/10.1038/s41592-026-03254-z)
- **Copyright/license:** © The Author(s) 2026; CC-BY-4.0.
- **License evidence:** Physical PDF page 5 grants Creative Commons Attribution
  4.0 International reuse and has the same article DOI in its header. The title
  and byline appear on page 1. The grant includes images unless separately credited
  otherwise; Figure 1 has no exclusion credit.
- **PDF:** `../pdfs/schreiber-2026-tangermeme-a-toolkit-for-understanding.pdf`
- **SHA-256:** `19f22c33251482c7eba9f6e754980735ccdc03b44e0e4a5828ca6cc19c98aa6f`
- **Material used:** Figure 1 on physical page 2 (cases v13 and v14).
- **Changes:** Full page rendered to RGB PNG at 110 DPI; scientific content unchanged.

## Cell-identification comparison

- **Title:** A comparison of automatic cell identification methods for single-cell RNA sequencing data
- **Authors:** Tamim Abdelaal, Lieke Michielsen, Davy Cats, Dylan Hoogduin,
  Hailiang Mei, Marcel J. T. Reinders, Ahmed Mahfouz
- **Source:** Genome Biology 20:194 (2019),
  [doi:10.1186/s13059-019-1795-z](https://doi.org/10.1186/s13059-019-1795-z)
- **Copyright/license:** © The Author(s) 2019; CC-BY-4.0.
- **License evidence:** Physical PDF page 1 grants Creative Commons Attribution
  4.0 International reuse on the article's title/byline/DOI page. Figures 1 and 2
  carry no separate exclusion credit.
- **PDF:** `../pdfs/abdelaal-2019-a-comparison-of-automatic-cell.pdf`
- **SHA-256:** `89a626d7e6ba86e27c12230f697b7f15fcc996d359a505d376548f2f34ccf2d6`
- **Material used:** Figure 1 on physical page 5 and Figure 2 on page 7
  (cases v15 and v16).
- **Changes:** Full pages rendered to RGB PNG at 110 DPI; scientific content unchanged.

## DeepCRISPR

- **Title:** DeepCRISPR: optimized CRISPR guide RNA design by deep learning
- **Authors:** Guohui Chuai, Hanhui Ma, Jifang Yan, Ming Chen, Nanfang Hong,
  Dongyu Xue, Chi Zhou, Chenyu Zhu, Ke Chen, Bin Duan, Feng Gu, Sheng Qu,
  Deshuang Huang, Jia Wei, Qi Liu
- **Source:** Genome Biology 19:80 (2018),
  [doi:10.1186/s13059-018-1459-4](https://doi.org/10.1186/s13059-018-1459-4)
- **Copyright/license:** © The Author(s) 2018; CC-BY-4.0.
- **License evidence:** Physical PDF page 1 grants Creative Commons Attribution
  4.0 International reuse on the article's title/byline/DOI page. Figures 2 and 3
  carry no separate exclusion credit.
- **PDF:** `../pdfs/chuai-2018-deepcrispr-optimized-crispr-guide-rna.pdf`
- **SHA-256:** `da06e848ed7368244c93e8983b7d71d033a17cdcf4335b7b39fd85f02a32d72f`
- **Material used:** Figure 3 on physical page 9 and Figure 2 on page 6
  (cases v17 and v18).
- **Changes:** Full pages rendered to RGB PNG at 110 DPI; scientific content unchanged.

## scAnnotate

- **Title:** scAnnotate: an automated cell-type annotation tool for single-cell RNA-sequencing data
- **Authors:** Xiangling Ji, Danielle Tsao, Kailun Bai, Min Tsao, Li Xing, Xuekui Zhang
- **Source:** Bioinformatics Advances (2023), vbad030,
  [doi:10.1093/bioadv/vbad030](https://doi.org/10.1093/bioadv/vbad030)
- **Copyright/license:** © The Author(s) 2023; CC-BY-4.0.
- **License evidence:** Physical PDF page 1 grants Creative Commons Attribution
  reuse with the explicit `/by/4.0/` license URL on the title/byline/DOI page.
  Figures 3 and 6 carry no separate exclusion credit.
- **PDF:** `../pdfs/ji-2023-scannotate-an-automated-cell-type-annotation.pdf`
- **SHA-256:** `c5fd7baa3ec9c02661b1034ac396ec684cd500b605df623c15a5950cb339e141`
- **Material used:** Figure 3 on physical page 7 and Figure 6 on page 8
  (cases v19 and v20).
- **Changes:** Full pages rendered to RGB PNG at 110 DPI; scientific content unchanged.
