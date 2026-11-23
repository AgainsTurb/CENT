# Reference verification report

Derived at 2026-11-21T08:59:29+00:00 by `refs/render_report.py` from `overrides.json`, `verified_refs.json` and the shipped `paper/refs.bib`. Do not edit by hand; re-run the script.

Three provenance classes, kept distinct:

- **registry** -- a Crossref or arXiv title match at similarity >= 0.87.
- **full-text** -- the paper itself, or its official DOI/proceedings record, was retrieved.
- **rejected** -- a match that was examined and refused. A rejected record is never the
  identifier of a cited work; the cited work is listed separately under its own key.

Shipped: 42 entries in `paper/refs.bib`; rows below: 47 (union of shipped, registry and override keys).

| key | class | score | identifier | venue | year | in manuscript |
|---|---|---|---|---|---|---|
| boundink | full-text | - | arXiv:2604.02103 | arXiv preprint arXiv:2604.02103 | 2026 | yes |
| calligan | registry | 1.0 | 2005.12500v1 | arXiv | 2020 | yes |
| casia | registry | 1.0 | 10.1109/icdar.2011.17 | 2011 International Conference on Documen | 2011 | yes |
| cffont | registry | 1.0 | 10.1109/cvpr52729.2023.00185 | 2023 IEEE/CVF Conference on Computer Vis | 2023 | yes |
| cfg | registry | 1.0 | 2207.12598v1 | arXiv | 2022 | yes |
| crnn | registry | 1.0 | 1507.05717v1 | arXiv | 2015 | yes |
| ddpm | registry | 1.0 | 2006.11239v2 | arXiv | 2020 | yes |
| deep_hccr | unresolved | - | - | - | - | **no** |
| dgfont | registry | 1.0 | 2104.03064v2 | arXiv | 2021 | yes |
| difffont | full-text | - | arXiv:2212.05895 | arXiv preprint arXiv:2212.05895 | 2022 | yes |
| diffusionpen | registry | 1.0 | 10.1007/978-3-031-73013-9_24 | Lecture Notes in Computer Science | 2024 | yes |
| diffvg | registry | 1.0 | 10.1145/3414685.3417871 | ACM Transactions on Graphics | 2020 | yes |
| diffwriter | full-text | - | 10.1007/978-981-99-8141-0_7 | Communications in Computer and Informati | 2023 | yes |
| dit | registry | 1.0 | 10.1109/iccv51070.2023.00387 | 2023 IEEE/CVF International Conference o | 2023 | yes |
| dp_font | full-text | - |  | International Joint Conference on Artifi | 2024 | yes |
| dsd | full-text | - | arXiv:2008.11354 | arXiv preprint arXiv:2008.11354 | 2020 | yes |
| emuru | registry | 1.0 | 2503.17074v2 | arXiv | 2025 | yes |
| fid | full-text | - | arXiv:1706.08500 | Advances in Neural Information Processin | 2017 | yes |
| fontdiffuser | registry | 1.0 | 2312.12142v1 | arXiv | 2023 | yes |
| ganwriting | registry | 1.0 | 2003.02567v2 | arXiv | 2020 | yes |
| graves2013 | registry | 1.0 | 1308.0850v5 | arXiv | 2013 | yes |
| hanzi | full-text | - |  | - | 2024 | yes |
| hi_gan | registry | 1.0 | 10.1609/aaai.v35i9.16917 | Proceedings of the AAAI Conference on Ar | 2021 | yes |
| hisac | full-text | - | 10.1007/978-3-032-37490-5_6 | Lecture Notes in Computer Science | 2026 | yes |
| hwt | registry | 1.0 | 2104.03964v1 | arXiv | 2021 | yes |
| icdar2013zh | registry | 1.0 | 10.1109/icdar.2013.218 | 2013 12th International Conference on Do | 2013 | yes |
| inception | registry | 1.0 | 10.1109/cvpr.2016.308 | 2016 IEEE Conference on Computer Vision  | 2016 | yes |
| inkspire | full-text | - |  | International Conference on Learning Rep | 2026 | yes |
| layout_glyph | registry | 1.0 | 2410.02309v3 | arXiv | 2024 | yes |
| ldm | registry | 1.0 | 2112.10752v2 | arXiv | 2021 | yes |
| lffont | registry | 1.0 | 10.1609/aaai.v35i3.16340 | Proceedings of the AAAI Conference on Ar | 2021 | yes |
| lvgh | full-text | - | arXiv:2511.11119 | arXiv preprint arXiv:2511.11119 | 2025 | yes |
| metascript | registry | 1.0 | 2312.16251v1 | arXiv | 2023 | yes |
| mxfont | unresolved | 0.6494 | - | - | - | **no** |
| one_dm | registry | 1.0 | 10.1007/978-3-031-73636-0_24 | Lecture Notes in Computer Science | 2024 | yes |
| rcgan | registry | 1.0 | 10.31577/cai_2024_2_482 | Computing and Informatics | 2024 | yes |
| scfont | full-text | - | 10.1609/aaai.v33i01.33014015 | AAAI Conference on Artificial Intelligen | 2019 | yes |
| schneider1990 | unresolved | - | - | - | - | **no** |
| scriptvit | registry | 1.0 | 2511.18307v1 | arXiv | 2025 | yes |
| sdt | registry | 1.0 | 2303.14736v2 | arXiv | 2023 | yes |
| softras | registry | 1.0 | 1904.01786v1 | arXiv | 2019 | yes |
| stroke2font | full-text | - |  | Algorithms | 2026 | yes |
| tortured | unresolved | 0.7253 | 10.36950/2024.9ciss010 | Current Issues in Sport Science (CISS) | 2024 | **no** |
| vatr | registry | 1.0 | 10.1109/cvpr52729.2023.02151 | 2023 IEEE/CVF Conference on Computer Vis | 2023 | yes |
| vatr_pp | registry | 1.0 | 2402.10798v1 | arXiv | 2024 | yes |
| wordstylist | registry | 1.0 | 2303.16576v2 | arXiv | 2023 | yes |
| zhangsuen | unresolved | - | - | - | - | **no** |

## Rejected matches (never cited as themselves)

- `fid` -- Crossref title search matched a title-squatting record (DOI 10.18034/ajase.v8i1.9, Asian Journal of Applied Science and Engineering) that is not Heusel et al.'s NeurIPS 2017 paper. Title similarity alone is not verification; the intended work is cited from its arXiv record (1706.08500) instead.
  - superseded by `arxiv:1706.08500`, which is the record listed for this citation in the table above.

## Coverage boundary

This report establishes bibliographic resolution and provenance class. It does not establish that a cited work supports the sentence citing it -- that is the integrity gate's Phase B, which is recorded in `phase2_5_integrity/integrity_report.md`.
