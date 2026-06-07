# Literature Reference Verification Report

**Date**: 2026-06-07
**Verifier**: Claude Code (deepseek-v4-pro)
**Source**: Cross-referencing `references.bib` against actual published DOIs via web search

---

## Summary

| Category | Count |
|----------|:-----:|
| References verified | 43 |
| Problems found | 11 |
| **DOI wrong** | 9 |
| **Authors wrong** | 3 |
| **Venue wrong** (journal vs conference) | 1 |
| **FATAL** (completely wrong reference) | 1 (Roelofs2024) |
| Chinese article needs replacement | 1 (Guo2025_HMM) |
| Unmatched PDF in lit dir | 1 (Rantanen) |
| References with no issues | 32 |

---

## Per-Reference Findings

### ✅ #4 Quddus2006 — DOI WRONG
- **Current**: `10.1016/j.trc.2006.04.003`
- **Correct**: `10.1016/j.trc.2006.08.004`
- **Note**: "08.004" not "04.003". Paper is correctly cited otherwise.

### ✅ #6 Hashemi2016 — DOI WRONG
- **Current**: `10.1080/15472450.2016.1178754`
- **Correct**: `10.1080/15472450.2016.1166058`
- **Note**: Middle digits "1178754" → "1166058". Journal, volume, pages are correct.

### ✅ #14 Qu2023 — DOI WRONG + AUTHORS WRONG
- **Current DOI**: `10.3390/ijgi12080310`
- **Correct DOI**: `10.3390/ijgi12080330`
- **Current authors**: "Qu, B. and Yang, S. and Qian, Y. and Wen, C."
- **Correct authors**: "Lin Qu, Yue Zhou, Jiangxin Li, Qiong Yu, Xinguo Jiang"
- **Note**: Both DOI (310→330) and author list are wrong. The actual paper has 5 authors, not 4 with different names.

### ✅ #20 Huang2024 — DOI WRONG + AUTHORS WRONG + MISSING METADATA
- **Current DOI**: `10.1109/TITS.2024.3421674`
- **Correct DOI**: `10.1109/TITS.2023.3314631`
- **Current authors**: "Huang, X. and Li, Y. and Wu, H. and Zhu, B."
- **Correct authors**: "Huang, Yulang and Wang, Dianhai and Xu, Wang and Cai, Zhengyi and Fu, Fengjie"
- **Missing**: Volume 25, Issue 2, Pages 1418–1429, Feb 2024
- **Note**: The DOI `2024.3421674` → `2023.3314631` — this is a completely different paper ID. Author list is entirely wrong.

### ✅ #24 Hansson2021 — DOI WRONG + VENUE WRONG + AUTHORS WRONG
- **Current DOI**: `10.1109/IV48863.2021.9575581`
- **Correct DOI**: `10.1109/TIV.2020.3035329`
- **Current venue**: `@inproceedings` — IEEE Intelligent Vehicles Symposium (IV)
- **Correct venue**: `@article` — IEEE Transactions on Intelligent Vehicles (TIV)
- **Current authors**: "Hansson, A. and Korsbacka, E. and Munthe, E. and Lind, H."
- **Correct authors**: "Anders Hansson, Ellen Korsberg, Roza Maghsood, Eliza Nordén, Selpi"
- **Note**: This is a JOURNAL article, not a conference paper. Entry type must change from `@inproceedings` to `@article`. Authors and DOI are completely wrong.

### ✅ #25 Lahrech2023 — DOI WRONG (page offset)
- **Current DOI**: `10.11591/ijece.v13i4.pp4072-4084`
- **Correct DOI**: `10.11591/ijece.v13i4.pp3924-3938`
- **Current pages**: 4072–4084
- **Correct pages**: 3924–3938
- **Note**: DOI and pages are offset. Journal title, authors, volume, issue are correct.

### ✅ #28 Maaref2022 — DOI WRONG + PAGES WRONG
- **Current DOI**: `10.1109/TITS.2021.3055201`
- **Correct DOI**: `10.1109/TITS.2021.3055200`
- **Current pages**: 5582–5594
- **Correct pages**: 5586–5601
- **Note**: DOI ends in "200" not "201". Page range is different.

### ✅ #29 AlHage2023 — DOI WRONG
- **Current DOI**: `10.1109/ITSC57777.2023.10422039`
- **Correct DOI**: `10.1109/ITSC57777.2023.10422598`
- **Note**: Last digits "2039" → "2598". Conference, authors, venue are correct.

### ✅ #34 Neamati2023 — DOI WRONG + TITLE INCOMPLETE
- **Current DOI**: `10.1016/j.artint.2023.103952`
- **Correct DOI**: `10.1016/j.artint.2023.104000`
- **Current title**: "Risk-aware autonomous localization with Mosaic Zonotope Shadow Matching"
- **Correct title**: "Risk-aware autonomous localization in harsh urban environments with mosaic zonotope shadow matching"
- **Missing**: Article number 104000, Volume 324
- **Note**: DOI article number "103952" → "104000".

### 🔴 #40 Roelofs2024 — FATAL: WRONG AUTHORS, VENUE, YEAR
- **Current**: Roelofs, Cain, Shlens, Mozer — JMLR Vol. 25, 2024
- **Actual paper with this title**: Chidambaram & Ge — ICLR 2025, arXiv:2406.04068
- **Resolution**: The title matches Chidambaram & Ge (2025) but the authors/venue are for a DIFFERENT paper. The Roelofs et al. JMLR paper exists but has a different title. 
- **Recommendation**: Replace with the correct reference:
  ```
  @inproceedings{Chidambaram2025,
    author = {Muthu Chidambaram and Rong Ge},
    title = {Reassessing How to Compare and Improve the Calibration of Machine Learning Models},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year = {2025},
    url = {https://arxiv.org/abs/2406.04068},
  }
  ```
  And update `\cite{Roelofs2024}` → `\cite{Chidambaram2025}` in the paper.

### ✅ #42 Bai2023 — DOI CORRECT
- **DOI**: `10.33012/2023.19458` — VERIFIED
- **Note**: The paper exists and DOI is correct. PDF just needs downloading. Authors: Xiwei Bai, Li-Ta Hsu. ION GNSS+ 2023, pp. 2258–2269.

---

## Additional Issues

### 🔴 #18 Guo2025_HMM — Chinese Article
- **Current**: Guo et al. 2025, 测绘通报 (Bulletin of Surveying and Mapping), in Chinese
- **User directive**: "USE ALL ENGLISH HIGH QUALITY ARTICLE"
- **Recommendation**: Remove from paper. Fu2021 (`~\cite{Fu2021}`) already covers second-order HMM map matching. The text `~\cite{Fu2021,Guo2025_HMM}` should become `~\cite{Fu2021}`. Delete the entry from references.bib.

### ⚠️ Unmatched PDF
- **File**: `Rantanen-2023-Open-geospatial-data-integration-in.pdf` in `docs/literatures/`
- **Status**: Not associated with any reference in `references.bib`
- **Recommendation**: Either add a bib entry if relevant, or delete the PDF file.

---

## Fixes Applied

All 11 reference errors have been corrected in `references.bib`. The Roelofs2024 key has been replaced with Chidambaram2025. Guo2025_HMM citation removed from paper text.
