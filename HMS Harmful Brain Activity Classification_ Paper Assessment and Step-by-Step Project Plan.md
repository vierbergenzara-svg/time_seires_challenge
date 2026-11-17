# HMS Harmful Brain Activity: What You Are Doing, Which Papers to Trust, and a Step-by-Step Plan (H0R29a, due 30/11/2026)

**Bottom line:** You are teaching a computer to copy how a panel of EEG experts votes on 10-second windows of brain activity from ICU patients. You are scored on how closely your predicted *distribution* of votes matches theirs (KL divergence, where lower is better).\[1\] Build a soft-label baseline with patient-grouped cross-validation on spectrogram band-power features first. Be suspicious of the papers claiming 94–99 % "accuracy". Anchor your writeup on Jing et al. 2023, ACNS 2021 and VIPEEGNet.

## TL;DR
- **What it is:** EEG recordings from critically ill patients are cut into 50-s windows. Between 1 and 28 experts voted on the central 10 s of each window, choosing one of six classes: Seizure, LPD, GPD, LRDA, GRDA, Other. The experts themselves agree only moderately (pairwise agreement 52 %, κ 42 %). The target is therefore a probability distribution, not one "right" answer. The best Kaggle teams reached a private KL of about 0.272; the course template's quick mode scores about 0.90.
- **Which papers to use:**
  - Use VIPEEGNet (Paper 5) as your main methods reference. It uses patient-grouped CV, the KL metric, and two-stage training on samples with many votes.
  - Use Classic ML / XGBoost on DEAP+HMS (Paper 2) to justify a baseline of hand-made features plus gradient boosting.
  - Use Dual-1D-CNN Feature Fusion (Paper 1) as the idea for an "advanced branch".
  - Do not compare your KL score to the 96–98 % accuracies of Papers 1, 2 and 4. Those numbers are almost certainly inflated by hard labels and window-level splits.
  - Paper 4 (Inception-v3 on Spectrograms) could not be found or verified.
- **What to do:**
  - Phase 1: three small notebooks (labels → signals → features + GroupKFold).
  - Phase 2: one notebook that grows from a class-prior baseline to LightGBM/XGBoost on band-power features with soft labels, then optionally one experimental branch.
  - Commit after every step; each step maps to a paragraph of the writeup.

## Key Findings

### (a) The five papers: self-reference titles and critical notes

| # | Self-reference title | What it actually is (verified) | One-line critical note |
|---|---|---|---|
| 1 | **Dual-1D-CNN Feature Fusion** | Unnisa et al., "EEG-based harmful brain activity classification using deep learning and feature fusion", *Scientific Reports* 16:20817 (2026). Raw EEG + MFCC + delta-1 + delta-2 into a dual 1D-CNN; 98.14 % accuracy, 99 % under 10-fold CV.\[2\]\[3\] | The main split is at the window level, so overlapping windows from the same recording end up in both train and test. A split by EEG ID still gives 97.84 %.\[2\] However, the EEG-ID counts per class (e.g. 169 seizure EEG-IDs for training) show a small curated subset, accuracy is measured against the consensus label, and KL divergence is not reported.\[4\] |
| 2 | **Classic ML / XGBoost on DEAP+HMS** | "Harmful Brain Activity Classification using EEG Signals", IEEE conference paper (IEEE Xplore 10837510). XGBoost, SVM, KNN, RF, DT; 99.2 % on DEAP, 96 % on HMS with XGBoost.\[5\] | DEAP is an *emotion* dataset, so it says nothing about seizures.\[5\] The 96 % on HMS is a hard-label accuracy with no KL reported.\[5\] Useful as precedent for "hand-crafted features + gradient boosting", not as a benchmark. This is very likely your [HMS] citation. |
| 3 | **cEEG Automation Review in Neurocritical Care** | Rubinos, Amorim, Rehim, Rodriguez-Ruiz, Zafar, Sheikh, Bruzzone et al., "Automating Critical Care EEG: Enhancing Accuracy, Efficiency, and Patient Outcomes in Critical Care" (SAGE, 2026; doi 10.1177/15357597261426958).\[6\] | A narrative review, not an experiment, so there are no methods to copy. Good for the background: the history from Berger 1924 → digital EEG → quantitative EEG → AI, integration with electronic medical records, and the line "the value of automation lies not in replacing clinical judgment, but in supporting it".\[6\] |
| 4 | **Inception-v3 on Spectrograms** | Could not be found. Neither I nor a dedicated search agent found a paper reporting 94.91 % with Inception-v3 on HMS. | Do not cite it until you have the PDF in hand. Similar HMS papers suggest its number would not be comparable to the competition. For example, TFCANet (*Sci Rep* 2026) reports 96.15 % on 5 classes with "Other" dropped, consensus hard labels, no patient-level split described and no KL.\[7\] |
| 5 | **VIPEEGNet** | Sun et al., "An Automated Classifier of Harmful Brain Activities for Clinical Usage Based on a Vision-Inspired Pre-trained Framework", *npj Digital Medicine* 8:768 (2025). 16-channel double-banana EEG → image → ImageNet-pretrained EfficientNetV2-B3; public/private KL 0.223/0.273, ranked 2nd vs 2,767 teams; 5-fold CV KL 0.225.\[8\]\[9\] | The most methodologically sound paper. Its folds are split by participant ("data from the same participant would not be split into different folds"). It uses the KL metric and a loss weighted by vote count, then fine-tunes on samples with ≥10 votes. Caveats: "ranked 2nd" is a comparison made after the competition, not an official placing, and the per-class recall (36.8–88.2 %) shows LRDA and GRDA remain hard.\[9\]\[10\] |

**Why accuracies of 96–99 % are implausible here.**
- In the companion interrater study, fellowship-trained experts agreed with each other only moderately (pairwise percent agreement 52 %, κ 42 %).\[11\]\[12\]
- In VIPEEGNet's high-quality subset, individual experts recovered the consensus seizure label with only 60.6 % recall, and LRDA with only 37.0 % precision.\[9\]
- A model that hits 98 % against the consensus would therefore be "better than the experts at agreeing with the experts". That almost always means one or more of the following:
  1. Splitting by window, so near-identical overlapping 50-s windows from the same eeg_id appear in both train and test.
  2. Collapsing soft votes into the `expert_consensus` hard label and discarding ambiguous (proto/edge) cases.
  3. Dropping the "Other" class or sampling a balanced, easy subset.\[7\]
- None of these papers reports KL on Kaggle's hidden test set, which is the only evaluation fully protected from leakage.

State this explicitly in your background. It is a strong, defensible critical point.

**The three papers to build on — and why.**
1. **VIPEEGNet.** It uses the competition's own metric, patient-grouped CV and the hidden test set. Its training recipe (loss weighted by vote count, then fine-tuning on samples with ≥10 votes) can be reused even in a model that is not deep learning.\[9\]
2. **Classic ML / XGBoost on DEAP+HMS.** It justifies the first real model you will build (features + gradient boosting). It matches the course's scikit-learn template and is already in your draft.
3. **Dual-1D-CNN Feature Fusion.** It shows the "raw EEG + engineered features" idea. It also contains a useful self-check (window-level vs EEG-ID-level split), which you can reproduce on a small scale to *demonstrate* leakage.\[2\]

These three form a natural ladder (features + trees → feature fusion → pretrained image model) and use the same dataset. The review (Paper 3) stays as a background citation. Paper 4 is dropped unless it can be verified.

**The citations that matter most for your background** (more than the five papers):
- Jing J, Ge W, Hong S, et al. "Development of Expert-Level Classification of Seizures and Rhythmic and Periodic Patterns During EEG Interpretation." *Neurology* 2023;100(17):e1750–e1762 (SPaRCNet; 6,095 EEGs from 2,711 patients; the model matched or exceeded most of 20 experts).\[13\]\[14\]\[15\]
- Jing J, Ge W, Struck AF, et al. "Interrater Reliability of Expert Electroencephalographers Identifying Seizures and Rhythmic and Periodic Patterns in EEGs." *Neurology* 2023;100(17):e1737–e1749 (doi 10.1212/WNL.0000000000201670).\[12\]
- Hirsch LJ, Fong MWK, Leitinger M, et al. "American Clinical Neurophysiology Society's Standardized Critical Care EEG Terminology: 2021 Version." *J Clin Neurophysiol* 2021;38(1):1–29.\[16\]
- The Kaggle competition itself (Jing J, Lin Z, Yang C, Chow A, Dane S, Sun J, Westover MB. "HMS – Harmful Brain Activity Classification", Kaggle, 2024).\[17\]\[18\]

## Details

### Q1. What is this, in plain words?
- **EEG** measures small voltage changes with electrodes on the scalp. Each electrode picks up the summed activity of millions of neurons beneath it, not single cells.\[19\]\[20\]
- **Why ICU patients are monitored:** after a stroke, brain injury, cardiac arrest, infection or sepsis, the brain can have seizures *without visible convulsions*.\[21\] These are called "non-convulsive" or electrographic seizures, and only EEG shows them. Seizures and seizure-like patterns "can harm the brain and contribute to in-hospital death, particularly when prolonged" (Jing et al. 2023).\[14\]
- **The ictal–interictal–injury continuum (IIIC):** between obvious seizures ("ictal") and normal brain activity lies a grey zone of rhythmic and periodic patterns.\[13\] Some are nearly seizures; others reflect injury without being seizures. The interrater study found that expert disagreements are best explained by each expert drawing the boundary at a different point on this continuum, not by careless reading.\[12\]
- **Why automation:** continuous ICU EEG produces days of 20-channel signal per patient, and there are too few specialists to read it in real time. The motivation in your draft is correct; both the review (Paper 3) and Jing et al. support it.\[22\]
- **What you are doing:** building a model that outputs six probabilities for each 10-s window.\[23\] These should match the share of experts who would choose each label.

### Q2. The six classes — and why you need them
| Class | Plain-language meaning | "Where" on the head |
|---|---|---|
| **Seizure (SZ)** | Sustained, evolving rhythmic abnormal activity — the brain "locked" into a firing pattern. | Anywhere |
| **LPD** – lateralized periodic discharges | Sharp "blips" repeating at regular intervals (e.g. ~1/s) with a flat gap between them, on one side. | One hemisphere |
| **GPD** – generalized periodic discharges | Same repeating blips, but on both sides at once, synchronous and symmetric. | Both hemispheres |
| **LRDA** – lateralized rhythmic delta activity | Slow (delta, <4 Hz) waves repeating with *no* gap between them, on one side. | One hemisphere |
| **GRDA** – generalized rhythmic delta activity | Same slow rhythmic waves on both sides. | Both hemispheres |
| **Other** | None of the above: normal background, artifacts, other abnormalities. | — |

**Naming logic (ACNS 2021):** *L* vs *G* tells you where the pattern is. *PD* vs *RDA* tells you its shape: periodic blips with gaps, or continuous rhythmic slow waves. ACNS 2021 cites Rodriguez Ruiz et al. (JAMA Neurol 2017; 4,772 patients), who found that LPDs, LRDA and GPDs "were associated with seizures, but generalized rhythmic delta activity was not".\[24\]\[25\]\[26\] This gives you useful clinical context for one background sentence.

**Does it matter for you? Yes, in three places:**
1. **The confusion matrix.** Expect confusions between L and G (LPD↔GPD, LRDA↔GRDA) and between PD and RDA.\[27\] The Kaggle overview shows edge cases split exactly like this.\[28\]
2. **Feature design.** Left/right asymmetry features directly encode the L-vs-G distinction.
3. **Background.** Two sentences defining the classes via ACNS.

### Q3. How the data were labeled
- Experts saw a 50-s EEG plus a 10-minute spectrogram centred on the same time, and labelled only the **central 10 s**.\[29\]\[30\]
- Each window has six vote counts (`seizure_vote` … `other_vote`). In train, the number of votes per window ranges from 1 to 28; test windows had 3–20 annotators.\[29\]\[30\]\[31\] The vote-count distribution has two peaks: many windows have 1–7 votes, others 10–28, and none have 8 or 9.\[20\]
- **Soft labels:** divide the votes by their sum to get a 6-number probability vector. For example, 3 GPD, 1 GRDA and 1 Other votes give 0.6/0.2/0.2. `expert_consensus` is just the class with the most votes, and is "provided for convenience only".\[30\]
- **Idealized / proto / edge** (Kaggle definitions):
  - *Idealized* = high agreement.
  - *Proto* = about half say "Other" and half say one named pattern (a partially formed pattern).
  - *Edge* = experts are split between two named patterns (e.g. LPD vs GPD).\[32\]\[33\]
- **Two subsets matter** (VIPEEGNet, Table 1):
  - 68,854 windows with 1–7 votes (1,794 patients).\[18\]
  - 39,946 windows with 10–28 votes (1,063 patients).\[18\]
  - Their class mix differs strongly: seizure is 29.5 % of the low-vote windows but only 1.5 % of the high-vote windows.\[9\]\[18\]
- The public leaderboard appears to behave like the high-vote subset.\[20\] The 15th-place team (Brendan Artley) reports that "training on data with >= 10 votes resulted in better LB scores". They also report that GroupKFold on this subset gave "a correlated CV/LB".\[34\]

### Q4. How the competition works
- **Not a puzzle.** It is a supervised prediction contest: you submit a notebook, Kaggle reruns it on a hidden test set, and it computes the score.\[1\]
- **The metric, KL divergence, in plain words:** for each window it asks "how surprised would the experts' vote distribution be by your prediction?"
  - Formula: KL(y‖ŷ) = Σ yᵢ·log(yᵢ/ŷᵢ), averaged over windows.\[20\]
  - 0 means a perfect match; larger is worse.
  - It punishes confidently wrong predictions severely. Predicting 0.0 for a class that got votes gives infinity, so always clip predictions (e.g. to ≥1e-6) and renormalize.
- **Baselines and reference scores:**
  - Predicting the uniform 1/6 on a window where all experts agree costs ln 6 ≈ 1.79.
  - Predicting the training-set class prior does somewhat better on average. Compute it yourself as your step-0 number.
  - The course template's quick mode (3,000 spectrograms) scored 0.9048 private / 0.8186 public.\[1\]
  - The winners (team Sony) reached 0.272332 private;\[20\] 2nd and 3rd scored 0.273765 and 0.274112.\[35\]
  - Ryan Peters' team placed 60th of 2,767 with 0.338933 private KL.\[20\]
  - My estimate (not a sourced figure) is that a realistic, good student result with features + gradient boosting lands in the 0.5–0.8 range.
- **Public vs private leaderboard:** the hidden test set is split in two. The public part is shown during the contest; the private part (~65 % of the test data) decides the final ranking.\[35\] Overfitting to the public board backfires.\[20\]
- **Status:** the competition closed in April 2024 with 2,767 teams.\[36\]\[37\] In practice, late submissions are still scored. The course template's README reports measured public/private scores and says the score "appears on the Submissions page after 10–20 minutes".\[1\] Late submissions are not placed on the official leaderboard. Your "ranking screenshot" will therefore be your Submissions page with public/private scores; compare it to the leaderboard to state where you would have ranked.
- **Code-competition rules:**
  - CPU or GPU notebook ≤ 9 hours run-time.
  - Internet access disabled.
  - External public data and pretrained models allowed (attached as Kaggle datasets).
  - Output file must be `submission.csv`.\[32\]

### Q5. Data format — what the numbers actually are
- **`train.csv`** (106,800 rows):
  - Columns: `eeg_id`, `eeg_sub_id`, `eeg_label_offset_seconds`, `spectrogram_id`, `spectrogram_sub_id`, `spectrogram_label_offset_seconds`, `label_id`, `patient_id`, `expert_consensus`, and six `*_vote` columns.
  - There are 17,089 unique EEG IDs, 11,138 spectrogram IDs and 1,950 patients.\[38\]\[39\]\[40\]
- **Raw EEG (`train_eegs/{eeg_id}.parquet`)** is a *table*:
  - Rows = time samples at **200 Hz**.\[18\]
  - Columns = 19 scalp electrodes in the 10-20 system (Fp1, F3, C3, P3, F7, T3, T5, O1, Fz, Cz, Pz, Fp2, F4, C4, P4, F8, T4, T6, O2) + **EKG** = 20 columns.\[18\]\[20\]
  - One labelled window = 50 s × 200 = **10,000 rows**, starting at `eeg_label_offset_seconds × 200`. The labelled centre is rows 4,000–6,000.\[41\]
  - Each file may hold a longer recording that contains many labelled windows.\[42\]
- **Spectrograms (`train_spectrograms/{spectrogram_id}.parquet`)** are also tables:
  - A `time` column + **400 columns** = 4 regions (LL, RL, LP, RP) × 100 frequency bins, named like `LL_0.59 … LL_19.92`.\[20\]\[31\]
  - Community analyses report a row every 2 s and frequencies of about 0.59–19.92 Hz. The "up to 20 Hz" range is confirmed; check the exact spacing in Exercise 2.\[20\]
  - One labelled 10-minute view = 300 rows, starting at `spectrogram_label_offset_seconds`.\[42\]
- **"88 numbers per recording" — correction:** nothing in the data has 88 columns. The real counts are:
  - 20 EEG columns (19 + EKG);
  - 16 or 18 bipolar "double-banana" channels, if you build them;
  - 400 spectrogram columns.
  If you saw 88 somewhere, it was probably a derived feature count in someone's notebook — check the source.
- **Are they in sync?** Yes. Both are centred on the same moment, and the 50-s EEG sits in the middle of the 10-minute spectrogram.\[43\] The spectrogram is a coarse, long, "zoomed-out" view (frequency × time); the raw EEG is the detailed short waveform.\[20\]
- **Sensors and regions:**
  - Each electrode mainly reflects the cortex below it. The letter gives the area: Fp = frontal pole, F = frontal, C = central, P = parietal, O = occipital, T = temporal.\[20\] Odd numbers are on the left, even numbers on the right, and z is the midline.
  - Clinicians view **bipolar** differences between neighbouring electrodes along four chains, called the "double banana":\[44\]
    - LL = Fp1-F7, F7-T3, T3-T5, T5-O1
    - LP = Fp1-F3, F3-C3, C3-P3, P3-O1
    - RP = Fp2-F4, F4-C4, C4-P4, P4-O2
    - RL = Fp2-F8, F8-T4, T4-T6, T6-O2
  - The four spectrogram regions are exactly these chains.\[20\]\[31\]
  - **Position matters by definition.** "Lateralized" vs "generalized" is half of each class name, so a model that averages away left vs right throws away the key signal.
- **IDs and leakage:**
  - One patient has several EEG recordings (`eeg_id`), and each recording has many overlapping labelled windows (`eeg_sub_id`), each with a unique `label_id`. Spectrograms are shared across windows.
  - Neighbouring windows overlap almost entirely and carry near-identical labels, so a random row split leaks.\[30\]
  - **Always split by `patient_id` (GroupKFold).** This is what VIPEEGNet does and what Papers 1, 2 and 4 apparently did not.\[9\]

### Q6. What top solutions did (for orientation only)
- **1st place (team Sony):**
  - Models on both raw EEG and spectrograms.
  - Two-stage training: stage 1 on windows with >1 vote, stage 2 on windows with >9 votes. Labels with a single vote were judged unreliable and excluded.
  - CV computed on one window per eeg_id with ≥10 votes.\[45\]
- **Common recipe across top teams:** KL-divergence loss, bipolar montage channels, band-pass filtering, robust normalization, and validation on high-vote samples.
  - The 4th-place team reports "The CV for 5939 data with vote_sum>=10 was 0.1983". One of its members pretrained on the full data and then fine-tuned "on clean data with vote≥10".\[46\]
  - The 7th-place team (Jebastin Nadar & Gunes Evitan) used 18 bipolar signal differences and reported "EKG signal gave no improvement". They dropped Kaggle's 10-minute spectrograms because these "consistently showed poor CV score".\[47\]
- **Vote weighting:** in Vikram Sandu's public GitHub write-up, multiplying the KL loss by log(total_votes + 1) "alone gave us a total of 0.02 boost in CV and 0.02 boost in LB".\[29\]
- **Takeaway for you:** the decisive ideas were about *labels and validation*, not giant models.\[48\] Those ideas are cheap enough to try in a student project.

### (c) GitHub template assessment
I could not open your fork (`vierbergenzara-svg/time_seires_challenge`) directly. The assessment below is based on the upstream course repo (`renaissanceee/time_seires_challenge`, 4 commits), which your fork appears to copy.\[1\]\[49\] Check that your fork has no extra files.

**What it provides:**
- `README.md` describing the task, the KL metric and how to run the code.\[1\]
- A scikit-learn baseline trained on **spectrograms**, in two forms: `hms_sklearn_baseline.ipynb` (to import into Kaggle) and `hms_sklearn_baseline.py` (the same code for local runs).\[1\]
- A `MODE` switch:
  - `quick` = random 3,000 spectrograms, ~5 min, private/public KL 0.9048/0.8186 (debugging only).\[1\]
  - `full` = all spectrograms, ~10 min, score "to be measured".\[1\]
- An optional cross-validation flag (`RUN_CV`).\[1\]
- A working Kaggle submission route: Accelerator = None, Internet = off, Save & Run All → Output → Submit.\[1\]
- `template.py`, `requirements.txt`, `.gitignore`, and instructions for local runs (Kaggle API download, ~30 GB of disk).\[1\]

**What it lacks (to add, in priority order):**
1. A documented full-mode score. Your first deliverable is filling in "to be measured".
2. Explicit patient-grouped CV. Confirm what `RUN_CV` does; if it is not `GroupKFold` on `patient_id`, replace it.
3. A trivial baseline (class prior) for context.
4. Any use of vote counts (sample weights or filtering to high-vote windows), and an explicit soft-label strategy.
5. Interpretable features that respect brain regions (band power per chain, left–right asymmetry).
6. Any use of the raw EEG parquet files.
7. Evaluation beyond one number: a per-class confusion matrix, CV on ≥10-vote windows, and a CV vs leaderboard comparison.
8. A results log (a table of experiment → CV KL → public/private KL) and figures for the writeup.

## Recommendations — the copy-paste plan for your LLM chat

```
=== PROJECT CONTEXT (paste into a fresh chat) ===
Course: AI for Biomedical Data Analysis (H0R29a), Kaggle time-series challenge.
Deadline 30/11/2026. Deliverables: Kaggle score screenshot, git commit history,
2-page writeup (1/2 p background, 3/4 p methods, 3/4 p results, + references).
I am a first-time ML student. Goal = understand and do clean, honest work, not copy winners.

Competition: HMS Harmful Brain Activity Classification (closed 2024; late submissions
are still scored on the Submissions page). Code competition: notebook <= 9 h,
internet OFF, output submission.csv.
Task: for each 10-s EEG window (centre of a 50-s window), predict 6 probabilities:
Seizure, LPD, GPD, LRDA, GRDA, Other. Target = expert vote distribution
(votes/sum). Metric = KL divergence (lower is better; clip preds >= 1e-6).
Reference points: class-prior baseline (compute), course template quick mode
~0.90 private, 1st place 0.272.
Data: train.csv (106,800 rows; 17,089 eeg_id; 1,950 patient_id; vote counts 1-28).
Raw EEG parquet: 200 Hz, 19 electrodes + EKG; window = 10,000 rows from
eeg_label_offset_seconds*200. Spectrogram parquet: time + 400 cols
(LL/RL/LP/RP x 100 freqs up to ~20 Hz); 10-min window = 300 rows from
spectrogram_label_offset_seconds.
Clinical key: L vs G = one side vs both sides; PD vs RDA = periodic blips with gaps
vs continuous slow rhythm. Experts only moderately agree (kappa ~0.42).
Rules for me: split by patient_id (GroupKFold) ALWAYS; never judge by accuracy
alone; one change at a time; commit after every step; log every score.
Starting code: my repo (fork of course sklearn baseline: spectrogram features,
MODE quick/full, RUN_CV).
Core papers: Jing 2023 Neurology (SPaRCNet + interrater), Hirsch 2021 ACNS,
VIPEEGNet (npj Digit Med 2025), Classic ML/XGBoost on DEAP+HMS (IEEE),
Dual-1D-CNN Feature Fusion (Sci Rep 2026).

=== PHASE 1: WHAT IS THE DATA (3 small notebooks, ~1 evening each) ===
Ex1  "01_labels.ipynb"  (WHY: you cannot model a target you don't understand)
  - Load train.csv. Count rows, unique eeg_id, patient_id, windows per patient.
  - Histogram of total votes (expect bimodal, nothing at 8-9).
  - Build soft labels; class distribution for <10 vs >=10 votes.
  - Label each window idealized / proto / edge; show 3 examples each.
  - Compute KL of (a) uniform, (b) class prior vs true labels -> first numbers.
  -> Writeup: background (labeling, soft labels), results (baseline row).
Ex2  "02_signals.ipynb"  (WHY: see what experts see; verify formats yourself)
  - Load one EEG parquet + one spectrogram parquet; print shape, columns, NaNs.
  - Cut the 50-s window and the 10-min spectrogram for one label_id.
  - Plot raw EEG (or double-banana bipolar chains) and the 4 spectrograms for
    1-2 high-agreement examples per class. Write one sentence per class on
    what you see (left vs right? blips vs waves?).
  - Check spectrogram time step and frequency range.
  -> Writeup: one figure for methods/background (example per class).
Ex3  "03_features.ipynb"  (WHY: turn signals into a table a model can learn from)
  - Clean: fill NaNs, log-transform spectrogram power, clip extremes.
  - Features per window: mean log-power in delta/theta/alpha/beta bands per
    region (4x4=16), same for the central 10 s vs full 10 min, left-right
    differences (LL-RL, LP-RP), simple stats (mean, std).
  - Targets: 6-column soft labels; weights = total votes.
  - GroupKFold(5) on patient_id; assert no patient in two folds.
  - Mini-experiment: random row split vs patient split with the same model
    -> show how leakage inflates the score (mirrors Paper 1's own check).
  -> Writeup: methods (features, CV), results (leakage demo = strong point).
DECISION POINT A (mine): one window per eeg_id, or all windows with group CV?

=== PHASE 2: ONE FINAL NOTEBOOK "hms_final.ipynb", grown step by step ===
Step 0  Copy the template; run MODE=full; submit; record CV + public/private.
        (WHY: a known-working end-to-end submission before changing anything)
Step 1  Add class-prior baseline + my KL function; check it matches template CV.
        (WHY: sanity check the metric; gives the floor to beat)
Step 2  Swap in my Phase-1 features + patient GroupKFold; model = LightGBM or
        XGBoost with soft labels (option: 6 regressors on probabilities then
        normalize, OR multiclass on duplicated rows weighted by votes).
        (WHY: Classic ML paper's approach, done with the correct metric/split)
Step 3  Use vote counts: (a) sample weights, (b) train all then refit or
        evaluate on >=10 votes only. Report CV on >=10-vote windows.
        (WHY: VIPEEGNet + 1st place: high-vote labels are more reliable and
        resemble the test set)
Step 4  Evaluate: per-class confusion matrix (argmax vs consensus), feature
        importances; explain confusions with L/G and PD/RDA logic.
        (WHY: turns a score into a biomedical result for the writeup)
Step 5  Final submission + screenshot; freeze; write results.
DECISION POINT B (mine): stop at Step 5, or try ONE branch below?

=== OPTIONAL BRANCHES (pick at most one; only after Step 5 works) ===
B1  KL-divergence loss + two-stage training in a small PyTorch MLP on the same
    features. Difficulty: low-medium. Ties to VIPEEGNet / top solutions.
B2  Small 2D CNN on the 4 spectrograms as an image. Difficulty: medium (GPU).
    Ties to Inception-v3 on Spectrograms idea.
B3  Pretrained EfficientNet on spectrogram/EEG images (attach weights as a
    Kaggle dataset; internet off). Difficulty: medium-high. Ties to VIPEEGNet.
B4  1D CNN on raw bipolar EEG (+ optional MFCC features). Difficulty: high.
    Ties to Dual-1D-CNN Feature Fusion.
B5  Add raw-EEG features (band power from Welch PSD on the 50-s / 10-s window)
    to the tree model. Difficulty: low-medium. "Fusion" without deep learning.
B6  Average predictions of 2 models (ensemble). Difficulty: low.
DECISION POINT C (mine): which branch, if any, given time and GPU quota?

=== OPEN QUESTIONS FOR MY REVISION ===
- Do I report CV on all windows, on >=10-vote windows, or both?
- Soft-label strategy for trees: regression-per-class vs weighted duplication?
- Do I keep Paper 2 as main related-work citation or replace it with Jing 2023?
- How many experiments fit before ~25/11 (leave 5 days for writing)?
```

**Practical Kaggle and git notes:**
- **Submitting:**
  - The final notebook must run end-to-end with Internet = off, read from `/kaggle/input/hms-harmful-brain-activity-classification/`, and write `submission.csv` (columns `eeg_id` + six `*_vote` probabilities).
  - Save Version → Save & Run All, then Output → Submit.\[1\]
  - Only a `test.csv` with one example is visible. The real test set is swapped in at scoring time, so never hard-code test IDs.\[1\]\[31\]
- **Pretrained weights (branch B3):** attach them as a Kaggle dataset, because internet is off during scoring.\[32\]
- **GPU:** the tree-model path needs no GPU (the template uses Accelerator = None).\[1\] Kaggle's GPU quota resets weekly and is about 30 hours, with sessions of up to 12 h (per Kaggle's GPU tips, as quoted in secondary guides).\[50\]\[51\] Save GPU time for one deep branch at most.
- **Git history:**
  - After each exercise or step, download the `.ipynb` (File → Download) into the repo and commit it with a descriptive message, e.g. "Ex3: patient GroupKFold + leakage demo" or "Final: LightGBM + vote weights, CV 0.xx, LB 0.xx".
  - Small, frequent commits are the evidence the graders want.
  - Do not commit the data; keep it in `.gitignore`.
- **Writeup mapping:**
  - Background ← Q1–Q3, plus Jing 2023 (both papers), ACNS 2021, Paper 3, and a critical paragraph on the accuracy claims of Papers 1/2/4.
  - Methods ← Ex3 + Phase 2 Steps 2–3 (features, soft labels, vote weighting, GroupKFold, KL).
  - Results ← a baseline table (uniform / prior / template / yours: CV, public, private), the leakage demo, the confusion matrix, and a one-sentence clinical interpretation.

**Suggested fixes to your current background draft:**
1. Keep the EEG and ICU sentences.
2. Add: "Seizures and seizure-like ictal–interictal–injury continuum patterns, standardized by the ACNS [Hirsch 2021], are associated with brain injury and in-hospital death [Jing 2023a]".
3. Describe the expert-voting dataset and the moderate interrater agreement [Jing 2023b].
4. Present [HMS]/Paper 2 with a caveat: its 96 % is a hard-label accuracy on a split not grouped by patient, so it is not comparable to the competition's KL metric. Contrast it with VIPEEGNet's patient-grouped KL of 0.225 (CV) / 0.273 (private test).\[38\]

## Caveats
- I could not read your fork or the code cells of the template notebook directly; the template description comes from the upstream course README. Check what `RUN_CV` actually splits on.
- Paper 4 (Inception-v3 on Spectrograms, 94.91 %) could not be found; treat it as unverified.
- The exact spectrogram time step (2 s) and frequency grid (0.59–19.92 Hz) are community-reported; confirm them in Exercise 2.
- That late submissions are scored is inferred from the course template's reported scores and Kaggle's general practice. I did not see an explicit statement on the competition page.
- The "0.5–0.8 realistic student range" is my estimate, not a sourced benchmark.
- VIPEEGNet's "ranked 2nd" compares a submission made after the competition against the frozen leaderboard; it is not an official placement. The paper also states its parameter savings inconsistently (0.7 % in the published abstract vs 2.8 % in the preprint).\[8\]\[10\]

## Sources

1. [GitHub - renaissanceee/time\_seires\_challenge: AI for Biomedical Data Analysis: Exercises (H0R29a) · GitHub](https://github.com/renaissanceee/time_seires_challenge)
2. [EEG-based harmful brain activity classification using deep learning and feature fusion](https://www.nature.com/articles/s41598-026-48670-5)
3. [EEG-based harmful brain activity classification using deep learning and feature fusion - PubMed](https://pubmed.ncbi.nlm.nih.gov/42092079/)
4. [EEG-based harmful brain activity classification using deep learning and feature fusion - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13338169/)
5. [Harmful Brain Activity Classification using EEG Signals](https://ieeexplore.ieee.org/document/10837510/)
6. [Automating Critical Care EEG: Enhancing Accuracy, Efficiency, and Patient Outcomes in Critical Care - Clio Rubinos, Edilberto Amorim, Erafat Rehim, Andres Rodriguez-Ruiz, Sahar F. Zafar, Zubeda B. Sheikh, Maria Jose Bruzzone, 2026](https://doi.org/10.1177/15357597261426958)
7. [A time-frequency cross-attention network model for epileptic seizure detection - Scientific Reports](https://www.nature.com/articles/s41598-026-41636-7)
8. [An Automated Classifier of Harmful Brain Activities for ...](https://www.nature.com/articles/s41746-025-02154-4.pdf)
9. [An Automated Classifier of Harmful Brain Activities for Clinical Usage Based on a Vision-Inspired Pre-trained Framework - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12715239)
10. [arxiv.org](https://arxiv.org/abs/2507.08874?context=cs)
11. [Improving Clinician Performance in Classification of EEG Patterns on the Ictal-Interictal-Injury Continuum using Interpretable Machine Learning](https://arxiv.org/pdf/2211.05207)
12. [Interrater Reliability of Expert Electroencephalographers Identifying Seizures and Rhythmic and Periodic Patterns in EEGs - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10136018/)
13. [Development of Expert-Level Classification of Seizures and Rhythmic and Periodic Patterns During EEG Interpretation - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10136013/)
14. [Development of Expert-Level Classification of Seizures and Rhythmic and Periodic Patterns During EEG Interpretation - PubMed](https://pubmed.ncbi.nlm.nih.gov/36878708/?dopt=Abstract)
15. [Development of Expert-Level Classification of Seizures and Rhythm...: Ingenta Connect](https://www.ingentaconnect.com/content/wk/wnl/2023/00000100/00000017/art00016?crawler=true&mimetype=application/pdf)
16. [American Clinical Neurophysiology Society's Standardized Critical Care EEG Terminology: 2021 Version - PubMed](https://pubmed.ncbi.nlm.nih.gov/33475321)
17. [(PDF) Harmful Brain Activity Classification of Spectrograms with Transfer Deep Learning](https://www.researchgate.net/publication/380025294_Harmful_Brain_Activity_Classification_of_Spectrograms_with_Transfer_Deep_Learning)
18. [An Automated Classifier of Harmful Brain Activities for Clinical Usage Based on a Vision-Inspired Pre-trained Framework](https://arxiv.org/pdf/2507.08874)
19. [Detection of epileptiform activity](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9579035)
20. [Harmful Brain Activity Classification](https://www.ryanirl.com/blog/hbac/)
21. [Wireless EEG unit](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10130278)
22. [Improving Clinician Performance in Classifying EEG Patterns on the Ictal–Interictal Injury Continuum Using Interpretable Machine Learning](https://ai.nejm.org/doi/full/10.1056/AIoa2300331)
23. [GitHub - limac246/brainActivity2024: Project on detecting harmful brain activity based on Kaggle project (Erdos Institute, Spring 2024)](https://github.com/limac246/brainActivity2024)
24. [2021 version](https://www.acns.org/UserFiles/file/ACNSNomenclature2021_MAIN-TEXT_2020_08-22ForPublicComment.pdf)
25. [Association of Periodic and Rhythmic Electroencephalographic Patterns With Seizures in Critically Ill Patients](https://jamanetwork.com/journals/jamaneurology/fullarticle/2593851)
26. [American Clinical Neurophysiology Society’s Standardized Critical Care EEG Terminology: 2021 Version - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8135051/)
27. [GitHub - mardhav-2008/ICU\_EEG\_Monitoring\_Assistant · GitHub](https://github.com/mardhav-2008/ICU_EEG_Monitoring_Assistant)
28. [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/c/hms-harmful-brain-activity-classification)
29. [GitHub - vikramsandu/HMS-Harmful-Brain-Activity-Classification: This repository contains a series of experiments that improved the classification performance of EEG-Spectrogram Data in the Kaggle competition HMS - Harmful Brain Activity Classification. · GitHub](https://github.com/vikramsandu/HMS-Harmful-Brain-Activity-Classification)
30. [HMS](https://www.kaggle.com/code/alejopaullier/hms-wavenet-pytorch-train/data)
31. [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data)
32. [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
33. [(Assignment) Machine Learning in Electroencephalogram & Spectrogram](https://medium.com/@65015079/assignment-machine-learning-in-electroencephalogram-spectrogram-e6ea404eaec6)
34. [GitHub - brendanartley/HMS-Competition: 15th place solution for the HMS - Harmful Brain Activity Classification Competition. · GitHub](https://github.com/brendanartley/HMS-Competition)
35. [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/leaderboard)
36. [HMS - Harmful Brain Activity Classification \[research, healthcare, signal processing, tabular, kullback leibler divergence\] - Standings - CLIST](https://clist.by/standings/hms-harmful-brain-activity-classification-research-healthcare-signal-processing-tabular-49077116/)
37. [Kaggle on X: "🎉 It's a wrap! The HMS - Harmful Brain Activity Classification competition where we challenged participants to classify seizures or harmful brain activity has closed. Congrats to the winners & thanks to all who competed. 👏 Overview of the competition + solution write-ups👇" / X](https://x.com/kaggle/status/1778884345844961321?lang=en)
38. [An Automated Classifier of Harmful Brain Activities for Clinical Usage Based on a Vision-Inspired Pre-trained Framework](https://www.nature.com/articles/s41746-025-02154-4)
39. [Wave2Word: A Multimodal Transformer Framework for Joint EEG-Text Alignment and Multi-Task Representation Learning in Neurocritical Care](https://arxiv.org/pdf/2601.00670)
40. [Comparative Analysis of Deep Learning Approaches for Harmful Brain Activity Detection Using EEG](https://arxiv.org/pdf/2412.07878)
41. [(PDF) An Automated Classifier of Harmful Brain Activities for Clinical Usage Based on a Vision-Inspired Pre-trained Framework](https://www.researchgate.net/publication/398835002_An_Automated_Classifier_of_Harmful_Brain_Activities_for_Clinical_Usage_Based_on_a_Vision-Inspired_Pre-trained_Framework)
42. [Home - HMS - Harmful Brain Activity Classification](https://sonujha.tech/posts/kaggle-Harmful-Brain-Activity-Classification/)
43. [GitHub - vikramsandu/HMS-Harmful-Brain-Activity-Classification: This repository contains a series of experiments that improved the classification performance of EEG-Spectrogram Data in the Kaggle competition HMS - Harmful Brain Activity Classification. · GitHub](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification)
44. [HMS - EEG Frequency Data](https://www.kaggle.com/datasets/frankling2021/hms-eeg-frequency-data)
45. [1st place solution, team Sony](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/writeups/team-sony-1st-place-solution-team-sony)
46. [4th place solution](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/writeups/aillis-go-bilzard-4th-place-solution)
47. [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/c/hms-harmful-brain-activity-classification/discussion/492281)
48. [Comparative Analysis of Deep Learning Approaches for Harmful Brain Activity Detection Using EEG](https://arxiv.org/html/2412.07878v1)
49. [GitHub - valentien430/time\_seires\_challenge: AI for Biomedical Data Analysis: Exercises (H0R29a) · GitHub](https://github.com/valentien430/time_seires_challenge)
50. [Kaggle Notebooks Free GPU: 30 Hours a Week, Session Limits and When to Move On (2026)](https://fastero.com/blog/kaggle-notebooks)
51. [Free Kaggle GPU: 2× NVIDIA T4, 16 GB Each (2026)](https://gabrubuilds.com/guides/kaggle-free-gpu/)
