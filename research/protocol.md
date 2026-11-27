# Experiment and report protocol

Every experiment under `research/` follows this protocol. It covers:
- how a piece of work is classified;
- how one experiment is designed;
- how its results are ordered;
- how they are reported;
- what is left out.

Register and wording follow `research/research-notebook.md` (copied from o4a-pe-paper
`.claude/output-styles/research-notebook.md`).

This copy is adapted to dot-PE from a protocol used in another repo, first for the row-limit
research (branch `research`) and then for the GPU project (branch `gpu-v2`). The adaptations are
the sections "Hands-off approval", "Data and compute", "GPU usage", "Engineering phases" and
"Gates and budget", and the opening requirements of a report (part 2 of "Report").

## Classification

Work is classified the way a GitHub pull request is, as a bug fix or a feature, or else as a measurement. The class decides what evidence the work needs.

A bug fix covers a result that is wrong and also one that is suboptimal. It has two experiments, in the same folder, `research/<question-slug>/`:
- The problem experiment shows either that the flaw exists in the current code or that it does not. Its report is complete without any fix and contains no fix.
- The solution experiment runs the problem experiment's test, unchanged, on the new code.

A feature is a capability the code does not have. It has no problem experiment. Its one experiment shows two things:
- that the feature is correct against an existing reference;
- how it performs, against that reference or against the existing option it would replace.

A measurement describes how existing code behaves, without claiming a flaw. It has one experiment and one report. A measurement that turns up a flaw does not become a problem report; the flaw becomes the question of a new problem experiment.

## One question

An experiment answers one question. The question is either a sentence with a yes or no answer, or a single measured quantity. A second question gets a second experiment.

## Design

The design is written in `design.md`, in the experiment's folder, and approved (see "Hands-off approval") before any result exists. It contains:

- the question, taken from the request;
- the prediction, the reason for it, and the outcome that would disprove it, but only if the request carries a hypothesis (a bug fix always does; a measurement usually does not, and then the design has no prediction rather than an invented one);
- a control, only if the question needs one: a case whose outcome could change how the result is read, such as a case where a suspected effect should be absent. A check of code nobody doubts, whose outcome is fixed by construction, is not a control and stays out of the report;
- the one variable that is swept, and everything held fixed;
- the measure, in words and with its units;
- a glossary of every term and symbol the report will use, each with its source in the code or in the papers;
- the figure chain, as a list of the questions the figures answer;
- the known confounders, and how each one is fixed.

The test code sits next to `design.md`. If the design changes after results exist, the change gets a dated note in `design.md` and a row in `progress.md`. The report describes only the experiment as it now is.

## Figure chain

Each figure answers one question. That question is the sentence immediately before the figure in the report.

A figure uses quantities that the setup or earlier figures defined, and it introduces at most one new quantity.

The chain moves from low level to high level and does not go back. A typical order is:
- one input;
- a quantity derived from it;
- the measure for one case;
- the measure across the swept variable.

Raw values are shown, as a scatter for example, before any summary statistic computed from them.

If there is a control, it comes first. The case under test follows, with the same sequence of figures and the same axes.

A figure that shows where an effect comes from, for example by splitting a sum into parts, comes after the effect itself has been shown. It is included only if the question needs it.

A figure that is not in the design's chain does not go in the report.

## Exploration

Exploratory notebooks go in `research/<question-slug>/explore/` or in `research/trash/`. Reports never cite them.

Something new found while exploring becomes a new question with its own design. It is not added to the current report.

## Report

The report is named `YYYY-MM-DD_problem.md`, `YYYY-MM-DD_solution.md`, `YYYY-MM-DD_feature.md` or `YYYY-MM-DD_measurement.md`. Its parts come in this order:

1. The purpose, in one to three plain sentences without specific numbers: what is tested and why it matters for the project. A reader who stops here knows what the report is about.
2. The rest of the opening, in one short paragraph: the dot-PE commit and branch the results were made with, the data source (the production rundir and the dot-PE commit that made it), and the report it supersedes as a bare link.
3. The question, and its answer in one sentence.
4. The prediction, quoted from `design.md`, and whether it held, if the design has one.
5. The setup, with every detail and configuration a reader in the field needs; it may be long, but it is never the entry point.
6. The figure chain. Each figure has its question before it and one or two sentences after it saying what it shows.
7. Observations after the experiment, if any: patterns noticed in the results that the question did not ask about. They are labelled as such, kept to a few sentences, carry no figure of their own, and nothing in the answer depends on them. One worth pursuing becomes its own question.
8. Limitations.
9. Reproduction.

A null result is reported the same way as a positive one.

Definitions and numbers:
- At its first use, and again at first use in each later section, a symbol gets its meaning in plain words in that same sentence, naming what is compared with what. Example: "Δ, the computed log-likelihood marginalized over phase and distance minus the exact value". The sentence links to the full definition.
- A complex quantity may have its full definition in the Setup. There it is a displayed equation on its own line, under a heading named for it, with each of its terms defined right there. It is never a clause inside a paragraph.
- Each number has its quantity, statistic, set and unit in the same sentence or in the table header.

## What to avoid

- History in a report: what was tried, revised or dropped before the experiment took its present form. That belongs in `design.md` notes and `progress.md`.
- Opening a report with the setup or with specific numbers instead of its purpose.
- Changing more than one thing between two runs that are compared.
- Choosing the measure, the test points or the range after seeing results, without saying so in the report.
- Editing the test between the problem report and the solution report.
- Showing a summary statistic before the raw values it summarises.
- Going back from a summary statistic to the single values, unless that step carries its own message. An example of such a message is which few values dominate the summary.
- Any figure, table or statistic outside the chain, including ones added "for completeness".
- Writing a hypothesis into the design or the report that the request did not contain, or presenting a pattern found in the results as if it had been the question.
- Presenting as findings effects below the resolution the user cares about.
- Fixing code inside a problem experiment.
- Using terms that neither the codebase nor its papers use.
- Using a symbol without defining it.
  - Each symbol is defined as "Report", "Definitions and numbers" says: its meaning in words at first use, and a full definition that a reader finds at once.
  - Each symbol keeps one meaning throughout the report.
  - The test: any term that would need an explanation in a seminar needs one in the report.
- Leaving a confounder that the design neither fixes nor names.
- Sanity checks shown as results: a figure whose outcome is fixed by construction tells the reader nothing about the question.

## Review

Before the user sees a report, a separate agent reads only `design.md` and the report. It checks that:

- every figure is in the design's chain, and in the same order;
- every term and symbol is defined at its first use;
- the prediction, if there is one, is quoted exactly as designed, and comes from the request;
- no claim goes beyond what the figures show.

A second separate agent then reads the report, with the papers and the code it cites, and has only one job:

- It lists every number and every term in the report.
- For each number, it quotes the quantity, statistic, set and unit the report gives in that sentence or in the table header.
- For each symbol, it quotes the meaning in words at first use in each section, and the full definition that sentence links to.
- It flags any number or symbol without these, a full definition that is not a displayed equation under its own heading, and any term found in neither the papers nor the code.

A report with any flagged item is fixed and checked again. It is not shown to the user until no item is flagged.

## Hands-off approval

The user is not available to approve each step. In place of "agreed with the user":

- A design is approved when it is committed and pushed on branch `gpu-v2`, a separate reviewer
  agent has read it against this protocol and the gates below, its findings are resolved, and the
  approval is a dated row in `research/progress.md`. No result of the experiment may exist before
  that row.
- Every decision that the user would otherwise have made (a gate passed or failed, a design
  changed, a direction dropped) is a dated row in `progress.md` with its reason. The user may
  revoke any of them later; nothing is merged into `main`.
- When a decision cannot be justified from the design and the gates, work on that question stops
  and the open decision is written at the top of `research/README.md`.

## Data and compute

- Production data is read in place and never written: the rundirs under
  `o4a-pe-paper/artifacts/pe/` and `o4b-pe-paper/artifacts/pe/`, and the shared `~/dot-PE`
  checkout. Outputs go to `artifacts/research/<question-slug>/` in this repo (git-ignored);
  symlinks are allowed where a script expects a local path.
- Anything heavier than a few minutes on one core runs through LSF (`bsub`), never on the login
  node. Every job is a row in `progress.md`: job id, submission time, purpose, and afterwards wall
  time, peak memory and exit status (`bhist -l`). `bjobs` and `bqueues` are checked before each
  submission. A local background task that is stopped is followed by a `ps` check for leftover
  processes.
- Full dot-PE runs use the `pe` profile of o4b-pe-paper `configs/lsf.yaml` (8 cores, 128000 MB,
  physics-medium) so their wall time is comparable with production. A rerun recycles the
  production run's incoherent stage and extrinsic samples and changes only the dot-PE version and
  the rundir.

## GPU usage

The cluster's GPUs are shared, and GPU time weighs heavily on fairshare (`short-gpu` has
`GPU_RUN_TIME_FACTOR` 10). Limits, without asking the user:

1. Queue `short-gpu` only. No `long-gpu`, `risk-gpu` or group-owned queues, and no idle
   interactive GPU sessions.
2. At most 1 GPU job running or pending at a time; 2 during the acceptance phase only. Always
   1 GPU per job.
3. Development jobs:
   - shared (`j_exclusive=no`), with `gmem` set to the measured need plus 20% and at most 40 GB;
   - `-W` at most 30 min;
   - at most 4 cores and 32 GB of host memory.

   Timing benchmarks:
   - exclusive;
   - `-W` at most 3 h;
   - at most 8 cores.
4. Every GPU job is preceded by a CPU run of the same entry point on a tiny input and by a
   `bjobs`/`bqueues` check. It is a row in `progress.md` with its submission time, GPU model,
   GPU-minutes and outcome. The rows keep a running total.
5. Budget: 60 GPU-hours for the project. Usage is reported to the user at 30 h. Going past 60 needs
   the user's approval.
6. Moving to a dedicated GPU cluster is reconsidered only if 5 consecutive jobs each wait more than
   2 h in the queue, or if the 6 h run limit blocks acceptance.

Submission goes through `research/lsf_submit.py --gpu MODEL --gmem GB [--exclusive]`.

`/home/projects/barakz/barakz/trex_gpu` (a collaborator's code) is read only. A global
`PreToolUse` hook (`~/.claude/hooks/trex_readonly.py`) enforces this. Its practices are copied
into our own scripts; their files and environment are never executed.

## Engineering phases

The GPU work has engineering phases (kernels, I/O, the end-to-end runner) that are loops of build,
measure and fix. Writing one design per microbenchmark would cost more than the benchmark. For
these phases:
1. one `design.md` per phase, written before the phase's code, fixes the phase's gate numbers;
2. microbenchmarks and correctness checks are rows in `progress.md`, without separate reports;
3. the phase ends with one short report that states the gate outcome.

The forensics measurement and the acceptance experiment follow the full protocol.

## Gates and budget

- CPU budget, without asking the user: at most 2 full CPU dot-PE reruns for the GPU project (only
  where acceptance needs a baseline that production does not provide); small jobs (1 core,
  minutes each) at most 20 at once. GPU budget: see "GPU usage".
- A full rerun needs a design that names the gate it passes, the expected wall time and peak
  memory (extrapolated from a measured pilot), and accept/reject numbers fixed before submission.
- GPU code lives in `dot-pe-gpu/dot_pe_gpu/`, with tests in `dot-pe-gpu/tests/`. Code in
  `dot_pe/` changes only where the GPU package needs a hook, and `pytest tests/` passes after any
  such change. New code comes with unit tests, and they pass before any run uses it.
- A failed acceptance test is reported as a null result. It is not re-tuned and rerun without a
  dated design note that says what changed and why.
