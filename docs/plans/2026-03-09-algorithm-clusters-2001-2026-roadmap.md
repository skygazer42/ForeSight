# ForeSight Algorithm Cluster Expansion 2001-2026 Roadmap

**Goal:** Build a year-aware and family-aware roadmap for expanding ForeSight with 50 additional algorithm clusters, with special attention to recurrent models, modern long-sequence models, graph/spatiotemporal forecasting, and foundation-model-era time-series systems.

**Scope:** This is a research and taxonomy document, not an implementation plan. It maps current local coverage, identifies gaps, and proposes a practical expansion queue.

**Status Note (March 9, 2026):** The first local/direct transformer-era parity wave and two lightweight families are now present in the registry: TFT, Informer, Autoformer, Non-stationary Transformer, FEDformer, TimesNet, iTransformer, TimeMixer, SparseTSF, LightTS, and FreTS.

---

## 1. Current Local Coverage Snapshot

### Primary local sources inspected

- `src/foresight/data/rnn_paper_metadata.json`
- `docs/rnn_paper_zoo.md`
- `README.md`
- `src/foresight/models/registry.py`

### What the repo already covers well

- Classic recurrent families through the paper zoo and wrapper zoo:
  SRN, LSTM/GRU variants, memory-augmented RNNs, attention-augmented RNNs,
  ConvRNN nowcasting variants, and many 2014-2018 recurrent ideas.
- Modern deep forecasting families already present as first-class models:
  PatchTST, Crossformer, Pyraformer, TSMixer, TimeMixer, LightTS, FreTS,
  RetNet,
  N-BEATS, NHITS, Mamba, RWKV, Hyena, ESRNN, LSTNet, TFT, Informer,
  Autoformer, FEDformer, Non-stationary Transformer, iTransformer,
  TimesNet, SparseTSF, and configurable xFormer variants.
- Several transformer-era models now have both global/panel and local/direct
  coverage: TFT, Informer, Autoformer, FEDformer, Non-stationary Transformer,
  iTransformer, and TimesNet.

### What the repo currently does not cover well

- Graph-native forecasting families with adjacency-aware data contracts.
- Foundation / pretrained / zero-shot time-series families.
- Diffusion and modern probabilistic generative forecasters.
- Continuous-time recurrent families such as LMU, LTC, and CfC.
- Structured state-space families beyond the current Mamba/RWKV/Hyena set.
- Post-2022 year coverage in the local RNN metadata is sparse.

---

## 2. Year Distribution In Existing Local RNN Metadata

The local RNN paper metadata contains **102 entries**, spanning **1986-2024**.

### Bucketed counts

| Year bucket | Entries |
| --- | ---: |
| pre-2001 | 5 |
| 2001-2005 | 3 |
| 2006-2010 | 1 |
| 2011-2015 | 28 |
| 2016-2020 | 58 |
| 2021-2026 | 7 |

### Exact counts for 2001-2026

| Year | Count | Notes |
| --- | ---: | --- |
| 2001 | 1 | Early reservoir / echo-state era |
| 2002 | 2 | Timing / liquid-state ideas |
| 2003 | 0 | Gap |
| 2004 | 0 | Gap |
| 2005 | 0 | Gap |
| 2006 | 0 | Gap |
| 2007 | 1 | Multi-dimensional recurrent modeling |
| 2008 | 0 | Gap |
| 2009 | 0 | Gap |
| 2010 | 0 | Gap |
| 2011 | 0 | Gap |
| 2012 | 1 | Transduction-style recurrence |
| 2013 | 0 | Gap |
| 2014 | 10 | Large jump: seq2seq / attention era starts |
| 2015 | 17 | Peak attention / memory / gated variants |
| 2016 | 14 | Multiscale / structured / spatiotemporal expansion |
| 2017 | 21 | Peak recurrent architecture exploration |
| 2018 | 17 | Strong spatiotemporal and efficiency work |
| 2019 | 3 | Noticeable slowdown |
| 2020 | 3 | Very light coverage |
| 2021 | 5 | Only a few new recurrent ideas |
| 2022 | 1 | Very sparse |
| 2023 | 0 | No local RNN metadata coverage |
| 2024 | 1 | Only `dg-rnn` in current metadata |
| 2025 | 0 | No local coverage |
| 2026 | 0 | No local coverage |

### Interpretation

- The local year curve is heavily concentrated in **2014-2018**.
- **2023 is empty**, even though this is exactly when foundation-model and
  time-series-LLM work accelerates.
- **2025-2026 are completely absent** in local metadata.
- The repo already documents many historical recurrent variants, but not the
  latest forecasting-centric clusters.

---

## 3. Major Category Clusters Already Present

### A. Recurrent and paper-zoo coverage

- Gated RNNs: LSTM, GRU, JANET, MGU, peephole, CIFG, Chrono, Phased, etc.
- Attention / memory recurrence: Bahdanau, Luong, NTM-style memory,
  pointer networks, CopyNet, neural stack/queue/RAM.
- ConvRNN / spatiotemporal recurrence: ConvLSTM, ConvGRU, TrajGRU,
  PredRNN, PredRNN++.

### B. First-class deep forecasting coverage

- Direct/local: PatchTST, Crossformer, Pyraformer, TSMixer, N-Linear,
  D-Linear, Mamba, RWKV, Hyena, ESRNN, CNN/TCN/ResNet1D/WaveNet.
- Global/panel: TFT, Informer, Autoformer, FEDformer, Non-stationary
  Transformer, iTransformer, TimesNet, PatchTST, Crossformer, Pyraformer.

### C. What is overrepresented vs underrepresented

- Overrepresented:
  paper-named recurrent variants from the 2014-2018 architecture boom.
- Underrepresented:
  graph forecasting, state-space lineages such as S4/S5, continuous-time
  recurrent models, diffusion models, and foundation-model families.

---

## 4. Fifty Additional Algorithm Clusters To Add

Legend for `status`:

- `implemented`: first-class family is already present in the registry.
- `missing`: no first-class family in the registry.
- `partial`: concept appears only in a paper-zoo baseline, or only in one
  interface where a fuller family should exist.

| # | Year | Cluster | Family | Status | Why it matters |
| --- | ---: | --- | --- | --- | --- |
| 1 | 2001 | Reservoir computing / ESN | recurrent | partial | Cheap long-memory baseline family; paper-zoo coverage exists but no dedicated ESN line |
| 2 | 2002 | Liquid-state / spiking reservoirs | recurrent | partial | Useful for event-driven and irregular dynamics |
| 3 | 2007 | Multi-dimensional / grid recurrent cells | recurrent | partial | Relevant to structured multivariate or spatial series |
| 4 | 2019 | Legendre Memory Unit (LMU) | recurrent | missing | Strong long-context recurrent inductive bias |
| 5 | 2020 | Liquid Time-Constant Networks (LTC) | recurrent | missing | Continuous-time sequence modeling with adaptive dynamics |
| 6 | 2022 | Closed-form Continuous-time (CfC) | recurrent | missing | Lightweight continuous-time recurrence, easier to train than LTC |
| 7 | 2021 | S4 | state-space | missing | Canonical modern long-sequence state-space family |
| 8 | 2022 | DSS / S4D | state-space | missing | Simpler diagonal SSM variants, easier first step than full S4 |
| 9 | 2022 | S5 | state-space | missing | Stronger expressive SSM follow-up to S4 |
| 10 | 2023 | RetNet / retention networks | long-sequence | implemented | Useful middle ground between attention and recurrence |
| 11 | 2024 | Mamba-2 / SSD refinements | state-space | partial | Repo has Mamba, but not the second-generation refinement line |
| 12 | 2024 | xLSTM | recurrent revival | missing | Important recurrent revival family with modern scaling story |
| 13 | 2024 | Griffin / Hawk recurrent hybrids | recurrent revival | missing | Strong modern recurrent alternatives to attention-heavy stacks |
| 14 | 2017 | STGCN | graph / traffic | missing | Canonical graph forecasting baseline for traffic and sensors |
| 15 | 2018 | DCRNN as a first-class graph family | graph / traffic | partial | Only lite paper-zoo presence today; no graph-native workflow |
| 16 | 2019 | Graph-attention forecasters | graph / traffic | missing | Covers ASTGCN / GMAN-style attention on sensor graphs |
| 17 | 2019 | Graph WaveNet | graph / traffic | missing | Still a very strong practical spatiotemporal baseline |
| 18 | 2020 | Graph-structure-learning forecasters | graph | missing | Learn adjacency instead of requiring a fixed graph |
| 19 | 2020 | AGCRN | graph / traffic | missing | Adaptive graph learning for dynamic relations |
| 20 | 2020 | MTGNN | graph / multivariate | missing | Widely used graph-temporal benchmark family |
| 21 | 2021 | StemGNN | graph / spectral | missing | Spectral graph-temporal forecasting for multivariate series |
| 22 | 2022 | STEP-style pretrained STGNNs | graph / pretraining | missing | Introduces graph pretraining as a reusable capability |
| 23 | 2022 | STID | graph / strong baseline | missing | Important surprisingly-strong lightweight baseline |
| 24 | 2023 | FourierGNN / spectral GNNs | graph / frequency | missing | Connects graph and frequency-domain forecasting |
| 25 | 2020 | TFT local/direct family | transformer / hybrid | implemented | Local/direct + global registry coverage now exists |
| 26 | 2021 | Informer local/direct family | transformer | implemented | Local/direct + global registry coverage now exists |
| 27 | 2021 | Autoformer local/direct family | transformer / decomposition | implemented | Local/direct + global registry coverage now exists |
| 28 | 2022 | Non-stationary Transformer local/direct | transformer | implemented | Local/direct + global registry coverage now exists |
| 29 | 2022 | FEDformer local/direct family | transformer / frequency | implemented | Local/direct + global registry coverage now exists |
| 30 | 2023 | TimesNet local/direct family | transformer / temporal 2D | implemented | Local/direct + global registry coverage now exists |
| 31 | 2024 | iTransformer local/direct family | transformer | implemented | Local/direct + global registry coverage now exists |
| 32 | 2022 | FiLM | decomposition / long-context | missing | Useful bridge between memory and spectral modeling |
| 33 | 2022 | LightTS | lightweight deep | implemented | Local/direct lightweight family is now present |
| 34 | 2023 | MICN | multiscale convolution | missing | Competitive multiscale convolutional decomposition family |
| 35 | 2023 | Koopa | decomposition / Koopman | missing | Brings Koopman-style dynamics into forecasting |
| 36 | 2023 | FreTS | frequency-domain MLP | implemented | Local/direct frequency-domain family is now present |
| 37 | 2024 | TimeMixer | mixer / decomposition | implemented | Local/direct mixer family is now present |
| 38 | 2024 | TimeXer | exogenous-aware transformer | missing | Important for future covariate heavy datasets |
| 39 | 2024 | SAMformer | linear-attention / adaptive mixing | missing | Modern efficient transformer-style family |
| 40 | 2024 | SparseTSF | sparse linear long-horizon | implemented | Local/direct sparse long-horizon baseline is now present |
| 41 | 2021 | TimeGrad | probabilistic / diffusion | missing | Canonical diffusion-style forecasting family |
| 42 | 2022 | TACTiS | probabilistic transformer | missing | Strong multivariate probabilistic forecasting line |
| 43 | 2023 | Lag-Llama | foundation / pretrained | missing | Open pretrained probabilistic forecasting family |
| 44 | 2023 | Time-LLM | LLM / reprogramming | missing | Important bridge between LLMs and forecasting |
| 45 | 2024 | TimesFM | foundation / zero-shot | missing | One of the most visible decoder-only forecasting foundation models |
| 46 | 2024 | Chronos / Chronos-Bolt | foundation / zero-shot | missing | Practical pretrained zero-shot family with strong ecosystem value |
| 47 | 2024 | Moirai / Moirai-MoE | foundation / universal forecasting | missing | High-value universal forecasting transformer family |
| 48 | 2024 | MOMENT | foundation / representation learning | missing | Useful for pretraining + transfer + embeddings |
| 49 | 2024 | Time-MoE | foundation / mixture-of-experts | missing | Represents the MoE branch of time-series foundation models |
| 50 | 2026 | Timer-S1 | foundation / reasoning-era TS model | missing | Good placeholder for 2025-2026 watchlist and latest-cycle coverage |

---

## 5. Year-First Program View

### 2001-2010: early recurrent and reservoir foundations

| Year bucket | Clusters | Family mix |
| --- | --- | --- |
| 2001-2005 | #1 Reservoir computing / ESN, #2 Liquid-state / spiking reservoirs | recurrent |
| 2006-2010 | #3 Multi-dimensional / grid recurrent cells | recurrent |

### 2011-2020: graph, continuous-time, and early modern forecasting gaps

| Year bucket | Clusters | Family mix |
| --- | --- | --- |
| 2017-2018 | #14 STGCN, #15 DCRNN as a first-class graph family | graph / traffic |
| 2019 | #4 LMU, #16 Graph-attention forecasters, #17 Graph WaveNet | recurrent, graph |
| 2020 | #5 LTC, #18 Graph-structure-learning forecasters, #19 AGCRN, #20 MTGNN, #25 TFT local/direct family | recurrent, graph, transformer |

### 2021-2022: state-space revival and transformer decomposition wave

| Year bucket | Clusters | Family mix |
| --- | --- | --- |
| 2021 | #7 S4, #21 StemGNN, #26 Informer local/direct family, #41 TimeGrad | state-space, graph, transformer, probabilistic |
| 2022 | #6 CfC, #8 DSS / S4D, #9 S5, #22 STEP-style pretrained STGNNs, #23 STID, #27 Autoformer local/direct family, #28 Non-stationary Transformer local/direct, #29 FEDformer local/direct family, #32 FiLM, #33 LightTS, #42 TACTiS | recurrent, state-space, graph, transformer, lightweight, probabilistic |

### 2023-2024: lightweight, foundation, and recurrent revival push

| Year bucket | Clusters | Family mix |
| --- | --- | --- |
| 2023 | #10 RetNet, #24 FourierGNN, #30 TimesNet local/direct family, #34 MICN, #35 Koopa, #36 FreTS, #43 Lag-Llama, #44 Time-LLM | long-sequence, graph, transformer, decomposition, frequency, foundation |
| 2024 | #11 Mamba-2 / SSD refinements, #12 xLSTM, #13 Griffin / Hawk recurrent hybrids, #31 iTransformer local/direct family, #37 TimeMixer, #38 TimeXer, #39 SAMformer, #40 SparseTSF, #45 TimesFM, #46 Chronos / Chronos-Bolt, #47 Moirai / Moirai-MoE, #48 MOMENT, #49 Time-MoE | state-space, recurrent revival, transformer, mixer, lightweight, foundation |

### 2025-2026: watchlist and reasoning-era placeholder

| Year bucket | Clusters | Family mix |
| --- | --- | --- |
| 2025-2026 | #50 Timer-S1 | foundation / reasoning-era |

## 6. Family-First Program View

### Recurrent / RNN and recurrent revival

- #1 Reservoir computing / ESN (2001)
- #2 Liquid-state / spiking reservoirs (2002)
- #3 Multi-dimensional / grid recurrent cells (2007)
- #4 LMU (2019)
- #5 LTC (2020)
- #6 CfC (2022)
- #12 xLSTM (2024)
- #13 Griffin / Hawk recurrent hybrids (2024)

### State-space and long-sequence sequence modeling

- #7 S4 (2021)
- #8 DSS / S4D (2022)
- #9 S5 (2022)
- #10 RetNet / retention networks (2023)
- #11 Mamba-2 / SSD refinements (2024)

### Graph and spatiotemporal forecasting

- #14 STGCN (2017)
- #15 DCRNN as a first-class graph family (2018)
- #16 Graph-attention forecasters (2019)
- #17 Graph WaveNet (2019)
- #18 Graph-structure-learning forecasters (2020)
- #19 AGCRN (2020)
- #20 MTGNN (2020)
- #21 StemGNN (2021)
- #22 STEP-style pretrained STGNNs (2022)
- #23 STID (2022)
- #24 FourierGNN / spectral GNNs (2023)

### Transformer, decomposition, mixer, and lightweight forecasting

- #25 TFT local/direct family (2020) - implemented
- #26 Informer local/direct family (2021) - implemented
- #27 Autoformer local/direct family (2021) - implemented
- #28 Non-stationary Transformer local/direct (2022) - implemented
- #29 FEDformer local/direct family (2022) - implemented
- #30 TimesNet local/direct family (2023) - implemented
- #31 iTransformer local/direct family (2024) - implemented
- #32 FiLM (2022)
- #33 LightTS (2022) - implemented
- #34 MICN (2023)
- #35 Koopa (2023)
- #36 FreTS (2023) - implemented
- #37 TimeMixer (2024) - implemented
- #38 TimeXer (2024)
- #39 SAMformer (2024)
- #40 SparseTSF (2024) - implemented

### Probabilistic, diffusion, and foundation-model families

- #41 TimeGrad (2021)
- #42 TACTiS (2022)
- #43 Lag-Llama (2023)
- #44 Time-LLM (2023)
- #45 TimesFM (2024)
- #46 Chronos / Chronos-Bolt (2024)
- #47 Moirai / Moirai-MoE (2024)
- #48 MOMENT (2024)
- #49 Time-MoE (2024)
- #50 Timer-S1 (2026)

## 7. Recommended Expansion Order

### Wave 0: already landed in the current expansion track

- #25 TFT local/direct family
- #26 Informer local/direct family
- #27 Autoformer local/direct family
- #28 Non-stationary Transformer local/direct family
- #29 FEDformer local/direct family
- #30 TimesNet local/direct family
- #31 iTransformer local/direct family
- #33 LightTS
- #36 FreTS
- #37 TimeMixer
- #40 SparseTSF

### Wave 1: next highest ROI, low conceptual risk

- Add API-compatible local/direct families that still fit the current lag-window
  interface:
  FiLM, MICN, Koopa, SAMformer.
- Add the covariate-aware local/global follow-up:
  TimeXer.

**Reason:** These are the next families that can reuse most of the current
registry, local training loop, and torch model patterns without introducing
graph data contracts or heavyweight external checkpoints.

### Wave 2: graph forecasting as a new product surface

- STGCN, DCRNN, Graph WaveNet, AGCRN, MTGNN, StemGNN, STID.

**Reason:** This is a coherent missing capability cluster and would require a
new graph-oriented data contract rather than one-off models.

### Wave 3: recurrent revival + state-space depth

- LMU, LTC, CfC, S4, DSS/S4D, S5, xLSTM, Mamba-2, Griffin/Hawk.

**Reason:** This materially modernizes the repo's recurrent story from
historical variants toward 2021-2024 sequence modeling.

### Wave 4: pretrained / zero-shot / foundation families

- Lag-Llama, TimesFM, Chronos, Moirai, MOMENT, Time-MoE, Timer-S1.

**Reason:** Highest user-facing differentiation, but requires decisions about
checkpoint management, inference-only wrappers, and possibly external model
weights.

---

## 8. Taxonomy Conclusions For A 2001-2026 View

If the user wants a **year-first taxonomy**, the cleanest split is:

1. **2001-2010**: reservoir and early structured recurrence
2. **2011-2015**: seq2seq, attention, memory augmentation, gated RNN variants
3. **2016-2020**: multiscale recurrence, spatiotemporal recurrence, graph traffic models
4. **2021-2022**: state-space revival, diffusion/probabilistic forecasting, decomposition transformers
5. **2023-2024**: foundation models, mixers, exogenous-aware transformers, recurrent revival
6. **2025-2026**: watchlist bucket for reasoning-era and MoE-style TS systems

### Practical takeaway

ForeSight already has a strong **history-heavy recurrent archive**.
The next 50 clusters should therefore skew toward:

- graph-native forecasting,
- structured state-space models,
- continuous-time recurrence,
- foundation / zero-shot families,
- and filling direct/local gaps for already-supported global transformer models.

---

## 9. Representative External References For The Missing New Wave

These are representative sources for the post-2022 expansion direction:

- Chronos: <https://arxiv.org/search/?query=Chronos%3A+Learning+the+Language+of+Time+Series&searchtype=all&source=header>
- Moirai: <https://arxiv.org/search/?query=Unified+Training+of+Universal+Time+Series+Forecasting+Transformers&searchtype=all&source=header>
- TimesFM: <https://arxiv.org/search/?query=A+decoder-only+foundation+model+for+time-series+forecasting&searchtype=all&source=header>
- MOMENT: <https://arxiv.org/search/?query=A+Family+of+Open+Time-series+Foundation+Models&searchtype=all&source=header>
- TimeMixer: <https://arxiv.org/search/?query=TimeMixer%3A+Decomposable+Multiscale+Mixing+for+Time+Series+Forecasting&searchtype=all&source=header>
- TimeXer: <https://arxiv.org/search/?query=TimeXer%3A+Empowering+Transformers+for+Time+Series+Forecasting+with+Exogenous+Variables&searchtype=all&source=header>
- Time-MoE: <https://arxiv.org/search/?query=Time-MoE%3A+Billion-Scale+Time+Series+Foundation+Models+with+Mixture+of+Experts&searchtype=all&source=header>
- Timer-S1: <https://arxiv.org/search/?query=Timer-S1%3A+Open+Visual-Time-Series+Forecasting+with+S1+Reasoning&searchtype=all&source=header>
