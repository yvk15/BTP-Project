# CLAUDE.md — project context for Claude sessions

This is Yash's personal copy of the team BTP repo (`yvk15/BTP-Project`, copied from
`vinay11hariharan-bit/BTP-Project`). The notes below were written by Yash in claude.ai chats and
pasted in on 2026-11-15. The repo itself (README.md, RTL/README.md, docs/) is the team's latest
written record; where the two disagree, see "Notes vs repo" first.

## Notes vs repo: conflicts to resolve before trusting either

Checked on 2026-11-15 against commit d448cb6 (team `main`). Not yet resolved with the user.

1. **Gestures, merge vs drop.** Notes: 12 FORS-EMG gestures merged into 6 output groups gave ~91%.
   Repo README (latest "merge vs drop" commit): the team uses 4 gestures (Hand Close, Hand Open,
   Wrist Flexion, Wrist Extension); merging 12 → 4 gave 60.9% vs 77.8% for dropping, and the
   earlier 6-group map gave 72.4% (−5.5 pp). The 91% figure does not appear in the repo.
2. **Where the LDA runs.** Notes: the classifier moved to the microcontroller; the chip only makes
   features. Repo RTL: `emg_lda_mac.v` is on the chip, and `RTL/README.md` describes a multi-board
   LDA core where boards pass K−1 partial sums along a chain and the head board does the argmax.
3. **Synthesis numbers.** Notes: 0.85 mm² at 24 electrodes (SKY130), 353,258 µm² for 8 electrodes.
   No area figures appear in the repo's markdown.
4. **NinaPro DB2 CNN study** (section A below): `btp_final_pipeline.py` is not in this repo. It lives in
   Colab/Drive.

Possible explanation (unconfirmed): the notes may describe Yash's own branch of the design and the
repo the team's. Ask the user which is current before building on either.

---

## How to work with this user

- Yash, EE @ IIT Bombay, 4th year. BTP team: Vinay Hariharan, Sudhindra Sahoo, Yashvardhan
  Khandegar, Digvijay Kumbhar. Advisor Prof. Laxmeesha Somappa (BioNICS group).
- Simple, natural English. Detailed technical explanations with relatable analogies. No fancy
  vocabulary, no generic AI intros, no repeating the same point.
- Explanation order: intuition (what and why) → concept → equations with every variable defined and
  its physical meaning → worked example → common mistakes → key takeaway. Never jump straight to a
  formula without its origin.
- Do not blindly agree. Correct wrong understanding respectfully, and point out wrong assumptions in
  the question itself.
- Accuracy over confidence. Never fabricate numbers, formulas or references. Never claim to have run
  something that wasn't run. Say clearly what is verified and what is not.
- Claude cannot see the real data (it is in Colab / Drive). Code can only be smoke-tested on
  synthetic data, so real accuracy stays unverified until the user pastes results back.
- Deliver complete, paste-ready code for one Colab cell or one script, with plots and tables, or
  step-by-step checkpoints where the output decides the next step.
- Ask the user to paste back the summary table, confusion matrix and per-subject results before
  deciding the next experiment. Never promise a target accuracy.
- Slides: flowchart-style over bullet points, a distinct colour palette per deck. Some reports are
  plain black-and-white.
- Professor wants controlled experiments, one variable at a time, not random changes.

## Goal

Control a low-cost prosthetic hand from forearm muscle signals (sEMG). A small chip extracts features
from the electrode signals; a cheap microcontroller decides the movement (see conflict 2). Long-term
target: low power, low latency, 8-bit hardware, so accuracy vs parameters vs MACs vs latency is a
real trade-off.

## Current direction

- Focus is on the LDA path with 5 integer features per electrode.
- Professor's two directions: (1) calibration, (2) multi-chip hardware.
- Professor feedback ("too little for four people") has pushed the project into an efficiency phase.

---

## A. Earlier study: subject-independent 1D CNN on NinaPro DB2 (history)

### Data
- NinaPro DB2, 10 subjects (S1–S10), 12 EMG channels, 2000 Hz. Label = `restimulus`, rest (0) excluded.
- Zip at `/content/drive/MyDrive/BTP/nina.zip`, extracted to
  `/content/nina/ninapro-dataset/DB2_sX/SX_E?_A1.mat`.
- **Known bug, fixed, do not reintroduce:** the original notebook loaded E1 + E2 + E3 together and
  kept labels 1, 6, 9, 13. E2 (23 gestures) and E3 (9 gestures) restart labels at 1, so "Gesture 1"
  mixed different movements. The final pipeline uses only `*_E1_*` files → 17 clean gestures,
  42,828 windows at 400 samples / 200 hop.
- Old mixed-label numbers (invalid, reference only): 4-class CNN ~55% on S9, ~71% on S10.

### Evaluation rules
- LOSO: train on 9, test on the 10th, ×10. Report mean ± std and a 95% t-interval CI
  (≈ ±9 points with n = 10).
- Tune on subjects 1–7 only. Subjects 8–10 are holdout, never used for a decision, reported separately.
- Fixed 15 epochs, cosine LR decay, no early stopping on the test subject.
- Select on balanced accuracy. Compare configs with paired per-subject Wilcoxon tests. Differences of
  1–3 points are noise, not results.
- Balanced class weights + label smoothing 0.1; AdamW (lr 1e-3, wd 1e-3); batch 64.

### Pipeline: `btp_final_pipeline.py` (Colab, single script split at `# %%`)
1. Load E1 only; per-subject per-channel z-score on that subject's active samples (label-free); clip ±10.
2. Visualise: gesture counts, example windows, channel-RMS patterns, PCA / t-SNE by gesture vs subject.
3. Gesture selection: LDA on 84 handcrafted features (MAV, RMS, VAR, WL, ZC, SSC, WAMP × 12 ch),
   LOSO, score all 2,380 four-gesture sets from one 17-class model.
4. Verify top-3 sets (+ original set) with the CNN under LOSO.
5. Depth sweep, 2 → 8 conv layers.
6. Greedy ablations: global vs subject normalisation, augmentation, input (raw / envelope /
   rectified), window length. Adopt only if > 0.5% over current best tune score.
7. Final: 3-seed ensemble + AdaBN, 95% CIs, paired tests, per-subject CSV, training curves,
   deployable model trained on all subjects.

- Cache: `/content/drive/MyDrive/BTP/btp_loso_cache.pkl`. Keys include gestures, depth, window, input,
  norm, aug, seeds, epochs, so keep EPOCHS = 15 to reuse it.
- Runtime: CPU so far; one LOSO run = 20–90 min. A T4 GPU is strongly recommended.

### Model
- 4 conv layers (professor's minimum): 16, 32, 48, 64 filters; kernels 7, 5, 5, 3. BatchNorm + ReLU;
  MaxPool after layers 1 and 3; SpatialDropout 0.15; global average pooling; Dense 64; Dropout 0.4;
  softmax(4).
- Input 600 × 12 (300 ms). 25,860 parameters, 5.27M MACs.
- Training-only augmentation: random channel gain, polarity flip, time shift (±40 samples), noise.

### Key findings
- 17-class LOSO with LDA features: 20.8% (chance 5.9%).
- Gesture choice matters most: 4-gesture sets range 28% → 70% (LDA). Every top-10 set has G9. Original
  set G1+G6+G9+G13 ranked 24/2380. Selected (on tune subjects): G7 + G9 + G13 + G14.
- Depth 2–8 all ~69–71% LOSO: capacity is not the bottleneck.
- Cross-subject shift is the real obstacle (t-SNE shows per-subject islands).
- Ablations vs baseline 70.8%: global norm 64.9% (p = 0.049); no augmentation 68.8% (p = 0.064);
  envelope input 67.8%; 300 ms window 72.7% (+1.9, p = 0.002) → adopted.
- Final CNN, LOSO ×10: single 72.7% (CI 63.7–81.7); 3-seed ensemble 72.2% ± 12.0 (CI 63.1–81.2);
  ensemble + AdaBN 75.9% (CI 68.4–83.3), +3.7, better in 8/10 subjects (p = 0.010).
  Holdout S8–S10 balanced accuracy 69.0% (72.6% with AdaBN).
- Per subject: best S5, S6 (~87%); worst S7 (~45%), S8 (~57%), high held-out loss → distribution shift.
- Per gesture recall: G9 81%, G7 73%, G13 74%, G14 49% (27% of G14 → G13).
- LDA + features baseline 67.4%; CNN not clearly better (p = 0.16–0.32).

### Caveats to state honestly
- Per-subject normalisation and AdaBN use unlabelled data from the new user: the claim is
  "subject-independent with unsupervised per-user calibration", not fully user-independent.
- 300 ms window adds latency (200 ms gave 70.8%).
- Decisions made on S1–S7, so the 10-subject mean is slightly optimistic; S8–S10 is the cleanest estimate.
- Only 10 subjects; one run per config except the final 3 seeds; no temporal smoothing.

### Open ideas (not done)
- Few-shot calibration with a gap between calibration and test windows (50% overlap leaks). Report
  as a different claim from user-independent.
- Within-subject upper bound; majority vote / smoothing across windows.
- Hardware: parameters, MACs, cycles, memory, 8-bit quantisation, BatchNorm folding.

---

## B. Main project: LDA classifier + feature-extraction chip (FORS-EMG)

### ML work
- 8 candidate models compared: nearly equal accuracy, very different hardware cost. LDA chosen.
- 5 integer features per electrode, no multipliers. Chip arithmetic costs no accuracy.
- Gestures: all 12 FORS-EMG gestures ~40%; merged into 6 output groups ~91%, one taught movement per
  group (see conflict 1).
- Calibration: short per-user calibration blending user + factory data adds ~10 points (~20 scored as
  6 groups). Recording only the 6 taught movements is +2.6 points on those and takes half the time
  (~2.5 min).
- Real-life tests: next-day use needs a daily 1-minute repetition blended with history; arm position
  and effort must be covered in calibration; rest = learned class + strength gate; confirm-4
  smoothing; shorter windows for less delay; checks on amputee data and NinaPro DB2 (~80% as deployed).
- CNN branch: calibrated CNN beat calibrated LDA by +2.4 points (94.3% vs 91.9%) at ~870× the
  arithmetic, below the agreed +3-point bar → LDA stays.

### Chip
- Review (Oct 30) of team v1 (no filter, older window, LDA on chip): 8 changes, 4 decisions, 8 things
  to keep, 5 fragile spots.
- Big split: classifier moved to the microcontroller; chip only makes features, so classifier changes
  become software updates (see conflict 2).
- Rebuilt in 6 checked steps: (1) 1 kHz sampling, settable banks; (2) signed 8-bit scaling; (3) SPI
  link to the MCU, every setting readable; (4) rest flag + 24-bit dead-electrode check; (5) 5-section
  filter with one shared multiplier; (6) calibration mode, loudness register, clean restart.
- Self-checks: dead/missing electrodes, missing ADS1299 pods, electrodes off skin, lost samples,
  impossible settings, warm-up, late or mixed frame reads.
- Accuracy kept: full chip within ±1 point of the ML model (−0.55 calibrated, +0.69 new user,
  FORS-EMG, 19 people).
- Checked: 41 test runs vs a bit-exact Python model; 6 real recordings through the Verilog with 0
  differences; MCU reference code matches every decision.
- Synthesised: 0.85 mm² at 24 electrodes, SkyWater 130 nm, 20,452 stored bits, no errors, one multiplier.

### Since the professor's feedback
- 8-electrode chip: 353,258 µm² once two memories are sized to the electrode count (−59%).
- Efficiency study: μ′ in 12 bits instead of 24 passes; leaky window, shared k per feature type and
  6-bit features lose accuracy → rejected; filter sections can't be judged on FORS-EMG (too clean),
  so the full filter stays.
- Proposed, not started: LDA → CNN cascade, CNN runs only when the LDA is unsure.

### Documents made
Decisions & Results Log; Story Repository (+ hardware part); 83-slide hardware deck; two-page
feature summary.

### Still open
- CNN-on-chip-samples check (script sent, results not back) and the cascade test script.
- Test with realistic disturbances before any filter change.
- FPGA: replay the dataset, then a live user with a real ADS1299 board.
- Power (not measured yet).
- Rest threshold on datasets with rest recordings.
- Amputee data (NinaPro DB3).
- Hand behaviour when an electrode fails.
