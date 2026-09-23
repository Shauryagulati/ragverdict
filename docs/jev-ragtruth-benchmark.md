# Jev vs LLM judges on RAGTruth

## Pre-registration

**Written:** 2026-09-22, before any judge scored a single RAGTruth *test* example.
**Status of this section:** frozen. Nothing below was edited after test predictions existed.
**How it is locked:** `bench/frozen_config.json` (sha256 below) and this section are in one local git commit made before the first test-split run; the benchmark CLI refuses every test-split run until `registered: true` is set in that commit. This commit was **not** pushed or publicly timestamped before the runs — readers should weigh that accordingly.


**Frozen config sha256:** `ff0ac1e8c8b523f17ee2e33787b7cfac3b6f95598aad967bffffd9cfff6f98cc` (`bench/frozen_config.json`).

### Question
On RAGTruth's human-labeled test set, how does **Jev** (TypeSafe AI's decision model, which returns a probability that a response is fully supported) compare with LLM-as-a-judge — **Claude Sonnet 5, DeepSeek Flash, GLM Flash** — and with a **cascade** (Jev first, Claude only when Jev is unsure) at detecting hallucinated RAG answers: accuracy, cost, speed, calibration, consistency, and where each breaks?

### Frozen choices (from 600 stratified *train* examples only)
| Setting | Value | Rule used |
|---|---|---|
| Jev model | `typesafe/jev-1.13` via OpenRouter (served `jev-1.13-20260917`) | pinned |
| Tuned Jev wording | **B**: "Does the response contain any information that contradicts the source or is not stated in the source?" (inverted → P(supported) = 1 − P(yes)) | highest train AUROC among A/B/C (A 0.9031, B 0.9064, C 0.9034 — effectively tied) |
| Tuned Jev threshold | **0.165** (hallucinated iff P(supported) < 0.165) | maximizes train macro-F1 (0.838). Deviation: the plan said "round to 2 decimals"; kept exact because rounding only moves off the optimum |
| Cascade band | **(0.065, 0.265)** exclusive | widest pre-listed band centred on the threshold where Jev's train accuracy < 0.70 (0.663, n=104) |
| Untuned Jev (headline) | wording **A** ("Is every factual claim in the response supported by the source?") at **0.5** | never tuned |
| LLM judges | ragverdict's faithfulness rubric; hallucinated iff `score < 1.0` or `supported_claims < total_claims` | fixed a priori |
| Claude | `claude-sonnet-5`, thinking disabled, Batch API; temperature cannot be set (API rejects it) | product default |
| DeepSeek / GLM | `deepseek/deepseek-v4.1-flash` (thinking off, temp 0) via DeepInfra fp8 / `z-ai/glm-5.3-flash` (thinking on, effort low) via Parasail fp8; OpenRouter, provider pinned, strict JSON schema | decoding mirrors the 2026-09-19 pilot; the makers' own endpoints don't support strict JSON schema on OpenRouter, so the highest-precision endpoints that do were pinned |
| Dataset | ParticleMedia/RAGTruth @ `c103204b`, test split; primary cohort `quality == good` (2,675; 894 hallucinated); label = ≥1 span with `implicit_true == false` | fixed a priori |
| CIs | percentile bootstrap resampling whole **source documents** (1,000 resamples, seed 0); H1 uses a 90% CI | fixed a priori |

Train-set observation recorded before testing: at the default 0.5 cutoff Jev flags 27% (A) – 37% (B) of clean train answers.

### Primary hypotheses (confirmatory)
- **H1 — Jev ≈ Claude.** Untuned F1 (Jev-A @0.5 minus Claude) on the common intersection of all judges, paired source-grouped **90%** CI. *Equivalent* if the CI lies within ±0.03; *Jev worse/better* if entirely below −0.03 / above +0.03; otherwise *inconclusive*. **Prediction: inconclusive or Jev worse** (train shows heavy false alarms at 0.5).
- **H9 — the cascade barely helps.** Cascade F1 at the frozen band minus the best single judge's F1, paired 95% CI. *Confirmed* if the CI upper bound < 0.02; *refuted* if lower bound ≥ 0.02; else inconclusive. **Prediction: confirmed.**

### Exploratory (reported with CIs; not corrected for multiplicity; descriptive where n is small)
- H2 Jev's recall on numeric hallucinations, at matched false-positive rate, is lower than Claude's.
- H3 Subtle hallucinations are harder than evident ones for every judge (n=30 subtle → descriptive).
- H4 GPT-4-written hallucinations are hardest (n=40 → descriptive).
- H5 Jev degrades in the longest within-task length quartile.
- H6 Jev is overconfident in its extreme bins (observed rate outside predicted in P<0.1 or P>0.9).
- H7 Jev's AUROC varies by ≥0.03 across 6 wordings (A–E + the pilot's).
- H8 Jev's flip rate across 3 repeats ≈ 0; Claude's > 1% (Claude cannot be run at temperature 0).
- H10 When Jev and Claude agree, precision > 0.85.
- H11 Thinking-on Claude gains < 0.03 AUROC at ≥3× cost (n=150 → likely inconclusive).
- H12 Among confident judge-vs-human disagreements, ≥25% are label errors (blind human audit; confirmed only if ≥13/30). **May not be run** — depends on the author's audit.

### Sensitivity analyses (pre-specified)
Invalid/missing counted as wrong; excluding 120 convention-dependent rows (49 clean only via implicit-true spans, 71 hallucinated only via due-to-null spans); the 2026-09-19 pilot's cohort rules on QA (900 responses, any-span label).

### Disclosed before results
- These hypotheses were written after seeing the 2026-09-19 slavadubrov pilot's QA *test* results (cited, git 5e14270) and a 60-example *train* spike.
- The harness was built with Claude (Anthropic), and Claude is one of the judges. No affiliation with TypeSafe, Anthropic, DeepSeek or Zhipu.
- Jev is closed; it may have seen RAGTruth in training. Train-tune AUROC (0.90) will be reported next to test AUROC.

---

## Results

*Not yet run. Everything above this line was written and committed before any test-split prediction existed.*
