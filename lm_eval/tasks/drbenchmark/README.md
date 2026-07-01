# DrBenchmark

### Paper

DrBenchmark is a Language Understanding Evaluation benchmark for the **French
biomedical domain**, introduced in
[DrBenchmark: A Large Language Understanding Evaluation Benchmark for French
Biomedical Domain](https://arxiv.org/abs/2402.13432) (Labrak et al., 2024).
It gathers a diverse set of clinical / biomedical downstream tasks
(classification, question answering, POS, NER, NLI and STS) and reports a single
**macro-averaged** score so a model can be ranked across all of them at once,
even though the constituent tasks use different primary metrics.

### Citation

```bibtex
@inproceedings{labrak2024drbenchmark,
  title={{DrBenchmark}: A Large Language Understanding Evaluation Benchmark for French Biomedical Domain},
  author={Labrak, Yanis and Bazoge, Adrien and Dufour, Richard and Rouvier, Mickael and Morin, Emmanuel and Daille, B{\'e}atrice and Gourraud, Pierre-Antoine},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  year={2024},
  url={https://arxiv.org/abs/2402.13432}
}
```

### Groups, Tags, and Tasks

#### Groups

- `drbenchmark`: runs every DrBenchmark task expressible in the harness and
  reports the DrBenchmark-style single score as the unweighted (macro) mean of
  the per-task `acc` (`aggregate_metric_list` in `_drbenchmark.yaml`).

#### Tags

- `drbenchmark_qa`: biomedical multiple-choice question answering.
- `drbenchmark_cls`: biomedical text-classification tasks.

#### Tasks

Each variant matches the classification / QA setup used in the published
DrBenchmark evaluation (accuracy as the primary metric).

- `drbenchmark_frenchmedmcqa` — **Main** variant: multiple-choice medical QA over
  the FrenchMedMCQA pharmacy-exam questions (5 options, A–E).
- `drbenchmark_cas` — sentence-level negation/uncertainty classification on the
  CAS clinical-case corpus.
- `drbenchmark_essai` — sentence-level negation/uncertainty classification on the
  ESSAI clinical-trial-protocol corpus.
- `drbenchmark_morfitt` — medical-specialty classification of French PubMed
  abstracts (MorFITT).

The token-level (QUAERO/E3C/MANTRA-GSC NER, CAS/ESSAI POS) and regression
(CLISTER/DEFT-2020 STS) tasks from the paper are not included here: they do not
map onto the harness's `multiple_choice` / `generate_until` output types and
would need dedicated scorers. The macro-average helpers in `utils.py`
(`primary_metric`, `benchmark_macro_average`) support aggregating a heterogeneous
results payload should those variants be added later.

### Usage

```bash
lm_eval --model hf --model_args pretrained=gpt2 --tasks drbenchmark
```

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation?
    * [x] Yes, the DrBenchmark datasets are released on the Hugging Face Hub under the `Dr-BERT` organization.

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted? (`drbenchmark_frenchmedmcqa`)
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
