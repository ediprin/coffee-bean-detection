# Reference Allocation Matrix for Proposal Bab II

Purpose: prevent citation poverty and citation recycling across the thesis proposal while preserving a campus-style Bab II structure.

## Core principle

A strong Bab II should not repeatedly reuse the same 5–8 familiar papers. Each subsection must have its own source pool appropriate to its function.

References are routed by role:

1. **Domain / coffee evidence** — coffee standards, defect taxonomy, coffee inspection, coffee detection/classification studies.
2. **Foundational method sources** — original or canonical papers for object detection, YOLO, Fourier/FFT, fine-grained recognition, etc.
3. **Recent methodological sources** — recent primary papers showing current development of the concept.
4. **Adjacent-domain bridge sources** — white pepper, maize seed, aircraft fine-grained detection, adverse-weather detection, etc.; used only to justify methodological plausibility.
5. **Implementation / protocol sources** — YOLO26 paper/docs and repository protocol when describing the actual experimental system.

The same paper may appear in more than one chapter only when it supports a genuinely different claim. Repetition should be intentional, not because it is the easiest citation to reuse.

## Anti-recycling rules

- Do not use Hong et al. as a universal citation for coffee, YOLO, fine-grained recognition, preprocessing, and methodology. Hong is a **pivot**, not a substitute for the whole literature.
- Do not use Kesiman et al. as the only source for both taxonomy and fine-grained difficulty; pair it with other 15–20 class coffee studies.
- Do not use FE-YOLO as the only frequency reference; separate Fourier theory, frequency-aware preprocessing, and frequency-aware feature processing.
- Do not use Syauqi et al. as generic evidence for all preprocessing. It is specifically a CLAHE-based composite preprocessing + YOLO white-pepper analogue.
- A non-foundational empirical paper should normally carry one major argumentative role in Bab II and at most one secondary role.
- Foundational papers may legitimately recur where definitions depend on them, e.g. the original YOLO/YOLO26 paper or Fourier theory source.
- Never add citations merely to increase count. Every citation must support a concrete sentence or comparison.

## Target reference diversity by subsection

These are planning targets, not mechanical quotas.

| Bab II section | Main reference pool | Target distinct references | Reuse policy |
|---|---|---:|---|
| 2.1 Biji Kopi Hijau dan Cacat Fisik | SNI/SCA standards, coffee taxonomy/dataset papers, coffee physical-quality studies | 5–8 | Prefer standards + taxonomy papers; avoid Hong unless needed for modern detection context |
| 2.2 Inspeksi Mutu Konvensional dan Tantangan | manual grading/inspection, machine-vision coffee reviews, practical coffee inspection studies | 4–7 | Use different sources from 2.1 where possible |
| 2.3 Object Detection | canonical object-detection sources + classification/localization diagnosis | 4–6 | Coffee papers are not needed for basic theory |
| 2.4 YOLO | original/canonical YOLO family papers + 2–3 coffee YOLO applications | 5–8 | Hong may appear here once; pair with Gope/Adiwijaya/Bahy/Jundullah rather than repeating Hong alone |
| 2.5 YOLO26 | original YOLO26 paper, official documentation/configuration, possibly one comparative source | 2–4 | This is the one section where the same primary YOLO26 source may recur in Bab III |
| 2.6 Fine-Grained Object Detection | general FG recognition/FGOD sources + multiple coffee fine-grained studies | 7–10 | Rotate coffee evidence: Kesiman, Hebert, Jundullah, Samudra, Hu, Jiao, etc. |
| 2.7 Preprocessing untuk Object Detection | classical preprocessing + detection-driven enhancement + agricultural analogues | 7–10 | Syauqi, Chen, IA-YOLO, DENet, Retinexformer/WCTE as appropriate; do not rely on one paper |
| 2.8 Frequency-Domain Representation | Fourier theory, texture spectrum, wavelet/frequency-aware CV, detection applications | 8–12 | Separate theory sources from detector application sources |
| 2.9 Penelitian Terkait | strongest cross-section primary studies | 12–18 rows/studies | Prefer distinct papers; table should visibly demonstrate literature breadth |

Expected Bab II unique-reference pool after full drafting: roughly **35–50 distinct primary/authoritative references**, with more in the bibliography once Bab I and Bab III are included. This is a diversity target, not a requirement to inflate citation count.

## Planned routing of core coffee papers

| Paper | Primary home in Bab II | Optional secondary use | Avoid using as |
|---|---|---|---|
| Hong et al. 2026 | 2.4 YOLO coffee / 2.9 related work | 2.6 one supporting sentence on visually similar defects | general preprocessing or frequency theory source |
| Bahy & Rifai 2026 | 2.1 taxonomy context or 2.4 coffee YOLO / 2.9 | 2.6 class-wise disparity | generic YOLO theory source |
| Jundullah et al. 2026 | 2.6 fine-grained coffee detection / 2.9 | 2.4 coffee YOLO landscape | preprocessing evidence |
| Hebert & Alamsyah 2026 | 2.6 difficult subtle classes / 2.9 | 2.1 SCA-style taxonomy context | general SCA standard authority unless the standard itself is cited |
| Samudra & Rachmawati 2025 | 2.6 visual similarity evidence | 2.9 related work | general coffee-detection benchmark |
| Gope et al. 2024/2025 | 2.4 YOLO-family viability / benchmarking | 2.9 | fine-grained 15–20 class evidence |
| Kesiman et al. 2023 | 2.1 SNI taxonomy/dataset context or 2.6 granularity difficulty | 2.9 | object-detection performance evidence |
| Arwatchananukul et al. 2024 | 2.6 fine-grained/generalization | 2.9 | YOLO/object-localization evidence |
| Jiao et al. 2025 | 2.6 internal discriminative representation | 2.9 | object-detection metric evidence |
| Hu et al. 2025 | 2.6 subtle visual differences / few-shot recognition | 2.9 | bounding-box detection evidence |
| Lei et al. 2025 | 2.3/2.6 classification-localization distinction in coffee | 2.9 | AF2 validation |

## Planned routing of method-bridge papers

| Paper/group | Primary home | Notes |
|---|---|---|
| IA-YOLO | 2.7 | task-driven input preprocessing before YOLO |
| DENet | 2.7 | detection-driven enhancement and LF/HF decomposition |
| Syauqi white pepper | 2.7 and 2.9 | fixed agricultural preprocessing analogue; two classes |
| Chen maize seed | 2.7 and 2.9 | seed-defect analogue; preprocessing effect separable from detector optimization |
| FE-YOLO | 2.8.4 and 2.9 | learned Fourier preprocessing before YOLO; not angular and not parameter-free |
| Cao et al. | 2.8.3 | primary radial/angular spectral theory source |
| Zhang & Tan | 2.8.3 | angular/orientation texture signature support |
| Xu et al. AFAB/LFDet | 2.6 or 2.8.4 and 2.9 | fine-grained frequency mechanism bridge; aircraft domain |
| WTConv / FDConv | 2.8.4 | internal frequency-aware methods; useful as contrast with input preprocessing |
| TOOD / Wu / IoU-Net | 2.3 | classification-localization diagnosis; do not recycle into unrelated sections |

## Citation density guideline

A normal theory paragraph should usually contain at least one authoritative source, and a comparative/synthesis paragraph should generally triangulate with 2–4 sources when it combines findings from multiple studies. Definitions need fewer but stronger citations; claims about trends or research gaps need broader evidence.

Avoid both extremes:

- **citation poverty**: several technical paragraphs supported by one recurring paper;
- **citation dumping**: 8–10 references appended to a sentence without explaining what each contributes.

## Related-work table strategy

The campus example uses a final table with columns for author/year, index, focus, method/model, and contribution/gap. We keep that format but improve evidence diversity.

The final table should contain a balanced set from three streams:

- coffee-domain detection/fine-grained studies;
- preprocessing / agricultural analogues;
- frequency/fine-grained methodological studies;

and end with **Penelitian yang Diusulkan**.

Hong should be visible as the pivot paper, but it should not dominate the table or the prose.

## Audit before proposal export

Before generating the final proposal document, perform a reference-use audit:

1. count unique references per subsection;
2. count how many subsections each paper appears in;
3. flag non-foundational papers appearing in 3 or more Bab II subsections;
4. flag paragraphs with substantive technical claims but no primary citation;
5. flag sections whose citations are >50% dominated by one paper or one research group;
6. verify all numerical claims against full-text primary PDFs;
7. verify journal/conference index/quartile separately before putting it in Table 2.1.

The purpose is not maximal reference count. The purpose is **coverage, diversity, traceability, and correct assignment of each source to the claim it actually supports**.
