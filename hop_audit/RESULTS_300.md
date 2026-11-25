# Hop-trace run: results (300 MMQA dev questions, released MAMMQA prompts)

Analysis written 2026-10-30. Pre-registration: `PREREGISTRATION.md`. Raw run: `mammqa_trace/` (Kaggle `mammqa-hop-trace-300`, version 1).

## 1. The run

- 300 of 300 questions completed in 5.35 h on 2×T4, about 57 questions per hour, with zero errors and zero evidence truncation.
- 62 of 2,100 calls hit the output-token cap. There were no out-of-memory fallbacks.
- The predictions were frozen before gold was read (sha256 `8d88b92a…`). The downloaded file matches that hash.
- **EM is 28.7% and F1 is 0.35.** By type, EM is: Compare 38.8%, TableQ 30–33%, Compose 27.6%, Intersect 15.4%.
- **Content accuracy is about 38%.** That counts EM, plus answers that contain the gold but miss the EM format. EM hides roughly one in five correct answers.

## 2. Pre-registered test: verdict

Claim: *most wrong answers happen after the bridge information is already inside the pipeline.*

| | P = info-present errors / errors | 95% CI | verdict |
|---|---|---|---|
| Automatic, hop_audit v0.1 (frozen) | **0.737** | [0.672, 0.799] | would be SUPPORTED |
| R1 fuzzy off / R2 no aliases / R3 weak-gold guard off | 0.733 / 0.737 / 0.769 | all above 0.66 | robust to matching choices |
| **R4 validation: auto vs manual agreement** | **78.7%** (37/47) | threshold 80% | **below threshold → primary reported as unreliable** |
| Manual labels, 60-question R4 sample | **0.56** | [0.42, 0.70] | **INCONCLUSIVE** |
| Automatic, hop_audit v0.2 (bugs fixed) | 0.747 | [0.685, 0.809] | same problem as v0.1 |

**The claim is not established.** Under the pre-registered rules the result is *inconclusive*.

The manual labels are mine (an AI annotator). They need a human second annotator before they can appear in a paper; see §6.

## 3. Why the automatic metric overestimates

1. **Mention is not use.** Take the sampled questions whose hops the audit marked REACHED. In 8 of those 46 (17%), the agent only *listed* the bridge (e.g. inside a full filmography or list of logos, 5 cases) or explicitly *rejected* it (3 cases). String matching cannot see this.
2. **Content-correct answers counted as errors.** 11 of the 60 sampled "wrong" answers are right in content. They differ only by an alias ("pantaloon"/"Pantalone", "Shahada"/"shahādah"), by gold formatting (the gold is a whole bio sentence), or by morphology ("Gloves"/"glove", "smiles"/"smiling"). The audit labels 6 of these as composition or commitment failures.
3. **Two scorer bugs**, found during validation and fixed in v0.2. Neither changes P.
   - The en-dash was not treated as a separator, so "3–3" did not match "3-3".
   - 24% of agent outputs (431 of 1,800) have no `<answer>` tag. v0.1 then treated the whole output as the committed answer.

## 4. What the validated (manual) labels show

From 60 random non-correct questions: 48 genuine errors, 11 content-correct, 1 gold error.

| failure | share of errors | what it looks like |
|---|---|---|
| **Bridge never identified** | **21/48 (44%)** | Mostly visual entity identification: which poster, logo or photo is the one described. Image bridges are recalled 54% of the time, against 93% for table bridges and 91% for text. |
| **Bridge found, a later hop failed** | 16/48 (33%) | Final-hop extraction (9): counting faces or vehicles, hair colour, a text or table lookup after the bridge. Incomplete answer lists (4). Other (3). |
| **Right answer in hand, discarded by the aggregator** | **10/48 (21%)** | An agent at Stage I or II answered correctly; the final answer abstained or followed another agent. |
| Commitment failure | 1/48 | |

- **Hand-off loss is not a major failure source.** At the stage level, only 2.9% of hops are lost between Stage I and Stage II. In the strict per-channel measure, 14.5% are dropped (26 of 30 drops are on the table channel). Hand-off was the concern MAMMQA's design raised, and the data does not support it.
- **Aggregation headroom.** A perfect selector over the existing agent answers would add about **+12 points of content accuracy** (manual estimate; 95% CI roughly +7 to +20). Automatic bounds range from +0.7 (strict) to +17 (lenient), so this needs semantic answer matching. Simple majority votes recover nothing, because agents answer in full sentences.

## 5. What this means for the paper

The original thesis, "the bridges arrive and the answer is still wrong", is **not supported as stated**. A defensible, data-backed framing is:

> **Where multi-agent MMQA actually fails: a validated, hop-level decomposition.** (i) Visual bridge identification is the largest bottleneck. (ii) The aggregator discards a correct answer in about one error in five. (iii) EM hides about one correct answer in five. (iv) Inter-agent hand-off, which MAMMQA's design targets, is rarely the problem.

This is still an evaluation and analysis paper built on MMQA's gold intermediate answers, which is the open niche identified in the novelty check. It now rests on validated labels rather than string matching alone.

## 6. Next steps (in order)

1. **Human second annotator (needed for any claim).** Label the same 60 questions blind, using the sheet in `results_300/r4_cases.txt`, then compute inter-annotator agreement (Cohen's κ). About 2–3 hours.
2. **Scorer v0.3 for mention vs use.** Count a hop as *used* only if it appears in an agent's answer or sub-answer, or as the subject of a claim. Do not count list-only mentions. **Validate on a new random 60-question sample**, not the one used here, so the scorer is not tuned on its own test set.
3. **Two diagnostic interventions**, each about 5 GPU-hours on the same 300 questions:
   - **Entity-anchored images (the EAG notebook, already built).** Give image titles, to measure how much of the 44% visual-bridge bucket is perception.
   - **Answer-span aggregator.** Extract a short span per agent, then select. This measures how much of the aggregation headroom is reachable.
4. If 1 and 2 hold, **extend to 600 questions** with the same protocol. That is the pre-registered response to an inconclusive result.

## Files

- `results_300/analysis_300_v01.json`: the frozen pre-registered analysis. `analysis_300_v02.json`: the bug-fixed version.
- `results_300/labels_300_v0*.csv`: the automatic label for each question.
- `results_300/r4_sample.json`, `r4_cases.txt`, `r4_judgments.jsonl`, `r4_summary.txt`: the manual validation.
- `results_300/aggregation_baselines.json`: the selection baselines.
- `outputs_300_v01/`, `outputs_300_v02/`: full audit outputs (per question, per hop, summary, report).
- `analysis/`: the analysis scripts.
