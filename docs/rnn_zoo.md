# RNN Zoo (100)

This document enumerates the 100 **RNN Zoo** models registered under `torch-rnnzoo-*-direct`.

RNN Zoo is a compact combinatorial family:

- **20 bases** (paper-named recurrent cores)
- **5 wrappers/variants** (`direct`, `bidir`, `ln`, `attn`, `proj`)

## Usage

List models:

```bash
foresight models list --prefix torch-rnnzoo
```

## Notes

- These implementations are **lite baselines** under a unified *direct multi-horizon forecasting* interface.
- The repo enforces a **no built-in recurrent modules** rule (PyTorch RNN/GRU/LSTM and their Cell variants);
  all recurrent cores are manual scan/unroll.
- Reference links below include **DOI / arXiv / URL** when available, plus search links for quick verification.
- `implementation` links point into this repo’s source for fast code navigation.

## Base Index (20)

| base | description | paper_id | paper_title | year | DOI | arXiv | URL | implementation | Semantic Scholar | Crossref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `elman` | Elman RNN (Elman, 1990) | `elman-srn` | Finding Structure in Time | 1990 | https://doi.org/10.1207/s15516709cog1402_1 | https://arxiv.org/search/?query=Elman+RNN+%28Elman%2C+1990%29&searchtype=all&source=header | https://doi.org/10.1207/s15516709cog1402_1 | - | https://www.semanticscholar.org/search?q=Elman+RNN+%28Elman%2C+1990%29 | https://search.crossref.org/?q=Elman+RNN+%28Elman%2C+1990%29 |
| `lstm` | LSTM (Hochreiter & Schmidhuber, 1997) | `lstm` | Long Short-Term Memory | 1997 | https://doi.org/10.1162/neco.1997.9.8.1735 | https://arxiv.org/search/?query=LSTM+%28Hochreiter+%26+Schmidhuber%2C+1997%29&searchtype=all&source=header | https://doi.org/10.1162/neco.1997.9.8.1735 | - | https://www.semanticscholar.org/search?q=LSTM+%28Hochreiter+%26+Schmidhuber%2C+1997%29 | https://search.crossref.org/?q=LSTM+%28Hochreiter+%26+Schmidhuber%2C+1997%29 |
| `gru` | GRU (Cho et al., 2014) | `gru` | Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation | 2014 | https://doi.org/10.3115/v1/d14-1179 | https://arxiv.org/abs/1406.1078 | https://doi.org/10.3115/v1/d14-1179 | - | https://www.semanticscholar.org/search?q=GRU+%28Cho+et+al.%2C+2014%29 | https://search.crossref.org/?q=GRU+%28Cho+et+al.%2C+2014%29 |
| `peephole-lstm` | Peephole LSTM (Gers et al., 2002) | `peephole-lstm` | Learning Precise Timing with LSTM Recurrent Networks | 2002 | - | https://arxiv.org/search/?query=Peephole+LSTM+%28Gers+et+al.%2C+2002%29&searchtype=all&source=header | https://www.jmlr.org/papers/v3/gers02a.html | - | https://www.semanticscholar.org/search?q=Peephole+LSTM+%28Gers+et+al.%2C+2002%29 | https://search.crossref.org/?q=Peephole+LSTM+%28Gers+et+al.%2C+2002%29 |
| `cifg-lstm` | CIFG / Coupled LSTM (Greff et al., 2015) | `cifg-lstm` | LSTM: A Search Space Odyssey | 2015 | https://doi.org/10.1109/TNNLS.2016.2582924 | https://arxiv.org/abs/1503.04069 | https://doi.org/10.1109/TNNLS.2016.2582924 | - | https://www.semanticscholar.org/search?q=CIFG+%2F+Coupled+LSTM+%28Greff+et+al.%2C+2015%29 | https://search.crossref.org/?q=CIFG+%2F+Coupled+LSTM+%28Greff+et+al.%2C+2015%29 |
| `janet` | JANET / Forget-gate LSTM (van der Westhuizen & Lasenby, 2018) | `janet` | The unreasonable effectiveness of the forget gate | 2018 | - | https://arxiv.org/abs/1804.04849 | https://arxiv.org/abs/1804.04849 | - | https://www.semanticscholar.org/search?q=JANET+%2F+Forget-gate+LSTM+%28van+der+Westhuizen+%26+Lasenby%2C+2018%29 | https://search.crossref.org/?q=JANET+%2F+Forget-gate+LSTM+%28van+der+Westhuizen+%26+Lasenby%2C+2018%29 |
| `indrnn` | IndRNN (Li et al., 2018) | `indrnn` | Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN | 2018 | https://doi.org/10.1109/cvpr.2018.00572 | https://arxiv.org/abs/1803.04831 | https://doi.org/10.1109/cvpr.2018.00572 | - | https://www.semanticscholar.org/search?q=IndRNN+%28Li+et+al.%2C+2018%29 | https://search.crossref.org/?q=IndRNN+%28Li+et+al.%2C+2018%29 |
| `minimalrnn` | MinimalRNN (Chen, 2017) | `minimalrnn` | MinimalRNN: Toward More Interpretable and Trainable Recurrent Neural Networks | 2017 | https://doi.org/10.1101/222208 | https://arxiv.org/abs/1711.06788 | https://doi.org/10.1101/222208 | - | https://www.semanticscholar.org/search?q=MinimalRNN+%28Chen%2C+2017%29 | https://search.crossref.org/?q=MinimalRNN+%28Chen%2C+2017%29 |
| `mgu` | MGU / Minimal Gated Unit (Zhou et al., 2016) | `mgu` | Minimal Gated Unit for Recurrent Neural Networks | 2016 | https://doi.org/10.1007/s11633-016-1006-2 | https://arxiv.org/abs/1603.09420 | https://doi.org/10.1007/s11633-016-1006-2 | - | https://www.semanticscholar.org/search?q=MGU+%2F+Minimal+Gated+Unit+%28Zhou+et+al.%2C+2016%29 | https://search.crossref.org/?q=MGU+%2F+Minimal+Gated+Unit+%28Zhou+et+al.%2C+2016%29 |
| `fastrnn` | FastRNN (Kusupati et al., 2018) | `fast-rnn` | FastGRNN: A Fast, Accurate, Stable and Tiny Kilobyte Sized Gated Recurrent Neural Network | 2018 | https://doi.org/10.1007/s12021-018-9371-3 | https://arxiv.org/abs/1901.02358 | https://doi.org/10.1007/s12021-018-9371-3 | - | https://www.semanticscholar.org/search?q=FastRNN+%28Kusupati+et+al.%2C+2018%29 | https://search.crossref.org/?q=FastRNN+%28Kusupati+et+al.%2C+2018%29 |
| `fastgrnn` | FastGRNN (Kusupati et al., 2018) | `fast-grnn` | FastGRNN: A Fast, Accurate, Stable and Tiny Kilobyte Sized Gated Recurrent Neural Network | 2018 | https://doi.org/10.1007/s12021-018-9371-3 | https://arxiv.org/abs/1901.02358 | https://doi.org/10.1007/s12021-018-9371-3 | - | https://www.semanticscholar.org/search?q=FastGRNN+%28Kusupati+et+al.%2C+2018%29 | https://search.crossref.org/?q=FastGRNN+%28Kusupati+et+al.%2C+2018%29 |
| `mut1` | MUT1 (Jozefowicz et al., 2015) | `mut1` | An Empirical Exploration of Recurrent Network Architectures | 2015 | - | https://arxiv.org/search/?query=MUT1+%28Jozefowicz+et+al.%2C+2015%29&searchtype=all&source=header | https://proceedings.mlr.press/v37/jozefowicz15.html | - | https://www.semanticscholar.org/search?q=MUT1+%28Jozefowicz+et+al.%2C+2015%29 | https://search.crossref.org/?q=MUT1+%28Jozefowicz+et+al.%2C+2015%29 |
| `mut2` | MUT2 (Jozefowicz et al., 2015) | `mut2` | An Empirical Exploration of Recurrent Network Architectures | 2015 | - | https://arxiv.org/search/?query=MUT2+%28Jozefowicz+et+al.%2C+2015%29&searchtype=all&source=header | https://proceedings.mlr.press/v37/jozefowicz15.html | - | https://www.semanticscholar.org/search?q=MUT2+%28Jozefowicz+et+al.%2C+2015%29 | https://search.crossref.org/?q=MUT2+%28Jozefowicz+et+al.%2C+2015%29 |
| `mut3` | MUT3 (Jozefowicz et al., 2015) | `mut3` | An Empirical Exploration of Recurrent Network Architectures | 2015 | - | https://arxiv.org/search/?query=MUT3+%28Jozefowicz+et+al.%2C+2015%29&searchtype=all&source=header | https://proceedings.mlr.press/v37/jozefowicz15.html | - | https://www.semanticscholar.org/search?q=MUT3+%28Jozefowicz+et+al.%2C+2015%29 | https://search.crossref.org/?q=MUT3+%28Jozefowicz+et+al.%2C+2015%29 |
| `ran` | Recurrent Additive Network (Lee et al., 2017) | `ran` | Recurrent Additive Networks | 2017 | https://doi.org/10.1016/j.neunet.2017.07.008 | https://arxiv.org/abs/1705.07393 | https://doi.org/10.1016/j.neunet.2017.07.008 | - | https://www.semanticscholar.org/search?q=Recurrent+Additive+Network+%28Lee+et+al.%2C+2017%29 | https://search.crossref.org/?q=Recurrent+Additive+Network+%28Lee+et+al.%2C+2017%29 |
| `scrn` | SCRN / Structurally Constrained RNN (Mikolov et al., 2014) | `scrn` | Learning Longer Memory in Recurrent Neural Networks | 2014 | https://doi.org/10.1007/978-3-319-11179-7_1 | https://arxiv.org/abs/1412.7753 | https://doi.org/10.1007/978-3-319-11179-7_1 | - | https://www.semanticscholar.org/search?q=SCRN+%2F+Structurally+Constrained+RNN+%28Mikolov+et+al.%2C+2014%29 | https://search.crossref.org/?q=SCRN+%2F+Structurally+Constrained+RNN+%28Mikolov+et+al.%2C+2014%29 |
| `rhn` | Recurrent Highway Network (Zilly et al., 2017) | `rhn` | Recurrent Highway Networks | 2017 | https://doi.org/10.21437/interspeech.2017-429 | https://arxiv.org/abs/1607.03474 | https://doi.org/10.21437/interspeech.2017-429 | src/foresight/models/torch_rnn_zoo.py#L862 | https://www.semanticscholar.org/search?q=Recurrent+Highway+Network+%28Zilly+et+al.%2C+2017%29 | https://search.crossref.org/?q=Recurrent+Highway+Network+%28Zilly+et+al.%2C+2017%29 |
| `clockwork` | Clockwork RNN (Koutník et al., 2014) | `clockwork-rnn` | A Clockwork RNN | 2014 | https://doi.org/10.1145/2576768.2598358 | https://arxiv.org/abs/1402.3511 | https://doi.org/10.1145/2576768.2598358 | src/foresight/models/torch_rnn_zoo.py#L864 | https://www.semanticscholar.org/search?q=Clockwork+RNN+%28Koutn%C3%ADk+et+al.%2C+2014%29 | https://search.crossref.org/?q=Clockwork+RNN+%28Koutn%C3%ADk+et+al.%2C+2014%29 |
| `qrnn` | QRNN / Quasi-Recurrent Neural Network (Bradbury et al., 2016) | `qrnn` | Quasi-Recurrent Neural Networks | 2016 | https://doi.org/10.1016/j.neunet.2016.04.001 | https://arxiv.org/abs/1611.01576 | https://doi.org/10.1016/j.neunet.2016.04.001 | src/foresight/models/torch_rnn_zoo.py#L866 | https://www.semanticscholar.org/search?q=QRNN+%2F+Quasi-Recurrent+Neural+Network+%28Bradbury+et+al.%2C+2016%29 | https://search.crossref.org/?q=QRNN+%2F+Quasi-Recurrent+Neural+Network+%28Bradbury+et+al.%2C+2016%29 |
| `phased-lstm` | Phased LSTM (Neil et al., 2016) | `phased-lstm` | Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences | 2016 | https://doi.org/10.1109/ebccsp.2016.7605258 | https://arxiv.org/abs/1610.09513 | https://doi.org/10.1109/ebccsp.2016.7605258 | - | https://www.semanticscholar.org/search?q=Phased+LSTM+%28Neil+et+al.%2C+2016%29 | https://search.crossref.org/?q=Phased+LSTM+%28Neil+et+al.%2C+2016%29 |

## Variant Index (5)

| variant | description | paper_id | paper_title | year | DOI | arXiv | URL | implementation | Semantic Scholar | Crossref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `direct` | direct head (last hidden -> horizon) | - |  |  | - | - | - | - | - | - |
| `bidir` | bidirectional wrapper (Schuster & Paliwal, 1997) | `bidirectional-rnn` | Bidirectional recurrent neural networks | 1997 | https://doi.org/10.1109/78.650093 | https://arxiv.org/search/?query=bidirectional+wrapper+%28Schuster+%26+Paliwal%2C+1997%29&searchtype=all&source=header | https://doi.org/10.1109/78.650093 | src/foresight/models/torch_rnn_zoo.py#L890 | https://www.semanticscholar.org/search?q=bidirectional+wrapper+%28Schuster+%26+Paliwal%2C+1997%29 | https://search.crossref.org/?q=bidirectional+wrapper+%28Schuster+%26+Paliwal%2C+1997%29 |
| `ln` | LayerNorm head (Ba et al., 2016) | `layer-normalization` | Layer Normalization | 2016 | - | https://arxiv.org/abs/1607.06450 | https://arxiv.org/abs/1607.06450 | src/foresight/models/torch_rnn_zoo.py#L893 | https://www.semanticscholar.org/search?q=LayerNorm+head+%28Ba+et+al.%2C+2016%29 | https://search.crossref.org/?q=LayerNorm+head+%28Ba+et+al.%2C+2016%29 |
| `attn` | additive attention pooling (Bahdanau et al., 2015) | `bahdanau-attention` | Neural Machine Translation by Jointly Learning to Align and Translate | 2015 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 | https://arxiv.org/abs/1409.0473 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 | src/foresight/models/torch_rnn_zoo.py#L899 | https://www.semanticscholar.org/search?q=additive+attention+pooling+%28Bahdanau+et+al.%2C+2015%29 | https://search.crossref.org/?q=additive+attention+pooling+%28Bahdanau+et+al.%2C+2015%29 |
| `proj` | projection head (Sak et al., 2014) | `lstm-projection` | Long Short-Term Memory Based Recurrent Neural Network Architectures for Large Vocabulary Speech Recognition | 2014 | https://doi.org/10.21437/interspeech.2014-80 | https://arxiv.org/abs/1402.1128 | https://doi.org/10.21437/interspeech.2014-80 | src/foresight/models/torch_rnn_zoo.py#L906 | https://www.semanticscholar.org/search?q=projection+head+%28Sak+et+al.%2C+2014%29 | https://search.crossref.org/?q=projection+head+%28Sak+et+al.%2C+2014%29 |

## Model Index (100)

| model_key | base | variant | base ref | wrapper ref |
| --- | --- | --- | --- | --- |
| `torch-rnnzoo-elman-direct` | `elman` | `direct` | https://doi.org/10.1207/s15516709cog1402_1 | - |
| `torch-rnnzoo-elman-bidir-direct` | `elman` | `bidir` | https://doi.org/10.1207/s15516709cog1402_1 | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-elman-ln-direct` | `elman` | `ln` | https://doi.org/10.1207/s15516709cog1402_1 | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-elman-attn-direct` | `elman` | `attn` | https://doi.org/10.1207/s15516709cog1402_1 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-elman-proj-direct` | `elman` | `proj` | https://doi.org/10.1207/s15516709cog1402_1 | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-lstm-direct` | `lstm` | `direct` | https://doi.org/10.1162/neco.1997.9.8.1735 | - |
| `torch-rnnzoo-lstm-bidir-direct` | `lstm` | `bidir` | https://doi.org/10.1162/neco.1997.9.8.1735 | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-lstm-ln-direct` | `lstm` | `ln` | https://doi.org/10.1162/neco.1997.9.8.1735 | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-lstm-attn-direct` | `lstm` | `attn` | https://doi.org/10.1162/neco.1997.9.8.1735 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-lstm-proj-direct` | `lstm` | `proj` | https://doi.org/10.1162/neco.1997.9.8.1735 | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-gru-direct` | `gru` | `direct` | https://doi.org/10.3115/v1/d14-1179 | - |
| `torch-rnnzoo-gru-bidir-direct` | `gru` | `bidir` | https://doi.org/10.3115/v1/d14-1179 | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-gru-ln-direct` | `gru` | `ln` | https://doi.org/10.3115/v1/d14-1179 | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-gru-attn-direct` | `gru` | `attn` | https://doi.org/10.3115/v1/d14-1179 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-gru-proj-direct` | `gru` | `proj` | https://doi.org/10.3115/v1/d14-1179 | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-peephole-lstm-direct` | `peephole-lstm` | `direct` | https://www.jmlr.org/papers/v3/gers02a.html | - |
| `torch-rnnzoo-peephole-lstm-bidir-direct` | `peephole-lstm` | `bidir` | https://www.jmlr.org/papers/v3/gers02a.html | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-peephole-lstm-ln-direct` | `peephole-lstm` | `ln` | https://www.jmlr.org/papers/v3/gers02a.html | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-peephole-lstm-attn-direct` | `peephole-lstm` | `attn` | https://www.jmlr.org/papers/v3/gers02a.html | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-peephole-lstm-proj-direct` | `peephole-lstm` | `proj` | https://www.jmlr.org/papers/v3/gers02a.html | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-cifg-lstm-direct` | `cifg-lstm` | `direct` | https://doi.org/10.1109/TNNLS.2016.2582924 | - |
| `torch-rnnzoo-cifg-lstm-bidir-direct` | `cifg-lstm` | `bidir` | https://doi.org/10.1109/TNNLS.2016.2582924 | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-cifg-lstm-ln-direct` | `cifg-lstm` | `ln` | https://doi.org/10.1109/TNNLS.2016.2582924 | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-cifg-lstm-attn-direct` | `cifg-lstm` | `attn` | https://doi.org/10.1109/TNNLS.2016.2582924 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-cifg-lstm-proj-direct` | `cifg-lstm` | `proj` | https://doi.org/10.1109/TNNLS.2016.2582924 | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-janet-direct` | `janet` | `direct` | https://arxiv.org/abs/1804.04849 | - |
| `torch-rnnzoo-janet-bidir-direct` | `janet` | `bidir` | https://arxiv.org/abs/1804.04849 | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-janet-ln-direct` | `janet` | `ln` | https://arxiv.org/abs/1804.04849 | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-janet-attn-direct` | `janet` | `attn` | https://arxiv.org/abs/1804.04849 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-janet-proj-direct` | `janet` | `proj` | https://arxiv.org/abs/1804.04849 | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-indrnn-direct` | `indrnn` | `direct` | https://doi.org/10.1109/cvpr.2018.00572 | - |
| `torch-rnnzoo-indrnn-bidir-direct` | `indrnn` | `bidir` | https://doi.org/10.1109/cvpr.2018.00572 | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-indrnn-ln-direct` | `indrnn` | `ln` | https://doi.org/10.1109/cvpr.2018.00572 | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-indrnn-attn-direct` | `indrnn` | `attn` | https://doi.org/10.1109/cvpr.2018.00572 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-indrnn-proj-direct` | `indrnn` | `proj` | https://doi.org/10.1109/cvpr.2018.00572 | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-minimalrnn-direct` | `minimalrnn` | `direct` | https://doi.org/10.1101/222208 | - |
| `torch-rnnzoo-minimalrnn-bidir-direct` | `minimalrnn` | `bidir` | https://doi.org/10.1101/222208 | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-minimalrnn-ln-direct` | `minimalrnn` | `ln` | https://doi.org/10.1101/222208 | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-minimalrnn-attn-direct` | `minimalrnn` | `attn` | https://doi.org/10.1101/222208 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-minimalrnn-proj-direct` | `minimalrnn` | `proj` | https://doi.org/10.1101/222208 | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-mgu-direct` | `mgu` | `direct` | https://doi.org/10.1007/s11633-016-1006-2 | - |
| `torch-rnnzoo-mgu-bidir-direct` | `mgu` | `bidir` | https://doi.org/10.1007/s11633-016-1006-2 | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-mgu-ln-direct` | `mgu` | `ln` | https://doi.org/10.1007/s11633-016-1006-2 | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-mgu-attn-direct` | `mgu` | `attn` | https://doi.org/10.1007/s11633-016-1006-2 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-mgu-proj-direct` | `mgu` | `proj` | https://doi.org/10.1007/s11633-016-1006-2 | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-fastrnn-direct` | `fastrnn` | `direct` | https://doi.org/10.1007/s12021-018-9371-3 | - |
| `torch-rnnzoo-fastrnn-bidir-direct` | `fastrnn` | `bidir` | https://doi.org/10.1007/s12021-018-9371-3 | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-fastrnn-ln-direct` | `fastrnn` | `ln` | https://doi.org/10.1007/s12021-018-9371-3 | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-fastrnn-attn-direct` | `fastrnn` | `attn` | https://doi.org/10.1007/s12021-018-9371-3 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-fastrnn-proj-direct` | `fastrnn` | `proj` | https://doi.org/10.1007/s12021-018-9371-3 | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-fastgrnn-direct` | `fastgrnn` | `direct` | https://doi.org/10.1007/s12021-018-9371-3 | - |
| `torch-rnnzoo-fastgrnn-bidir-direct` | `fastgrnn` | `bidir` | https://doi.org/10.1007/s12021-018-9371-3 | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-fastgrnn-ln-direct` | `fastgrnn` | `ln` | https://doi.org/10.1007/s12021-018-9371-3 | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-fastgrnn-attn-direct` | `fastgrnn` | `attn` | https://doi.org/10.1007/s12021-018-9371-3 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-fastgrnn-proj-direct` | `fastgrnn` | `proj` | https://doi.org/10.1007/s12021-018-9371-3 | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-mut1-direct` | `mut1` | `direct` | https://proceedings.mlr.press/v37/jozefowicz15.html | - |
| `torch-rnnzoo-mut1-bidir-direct` | `mut1` | `bidir` | https://proceedings.mlr.press/v37/jozefowicz15.html | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-mut1-ln-direct` | `mut1` | `ln` | https://proceedings.mlr.press/v37/jozefowicz15.html | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-mut1-attn-direct` | `mut1` | `attn` | https://proceedings.mlr.press/v37/jozefowicz15.html | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-mut1-proj-direct` | `mut1` | `proj` | https://proceedings.mlr.press/v37/jozefowicz15.html | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-mut2-direct` | `mut2` | `direct` | https://proceedings.mlr.press/v37/jozefowicz15.html | - |
| `torch-rnnzoo-mut2-bidir-direct` | `mut2` | `bidir` | https://proceedings.mlr.press/v37/jozefowicz15.html | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-mut2-ln-direct` | `mut2` | `ln` | https://proceedings.mlr.press/v37/jozefowicz15.html | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-mut2-attn-direct` | `mut2` | `attn` | https://proceedings.mlr.press/v37/jozefowicz15.html | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-mut2-proj-direct` | `mut2` | `proj` | https://proceedings.mlr.press/v37/jozefowicz15.html | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-mut3-direct` | `mut3` | `direct` | https://proceedings.mlr.press/v37/jozefowicz15.html | - |
| `torch-rnnzoo-mut3-bidir-direct` | `mut3` | `bidir` | https://proceedings.mlr.press/v37/jozefowicz15.html | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-mut3-ln-direct` | `mut3` | `ln` | https://proceedings.mlr.press/v37/jozefowicz15.html | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-mut3-attn-direct` | `mut3` | `attn` | https://proceedings.mlr.press/v37/jozefowicz15.html | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-mut3-proj-direct` | `mut3` | `proj` | https://proceedings.mlr.press/v37/jozefowicz15.html | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-ran-direct` | `ran` | `direct` | https://doi.org/10.1016/j.neunet.2017.07.008 | - |
| `torch-rnnzoo-ran-bidir-direct` | `ran` | `bidir` | https://doi.org/10.1016/j.neunet.2017.07.008 | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-ran-ln-direct` | `ran` | `ln` | https://doi.org/10.1016/j.neunet.2017.07.008 | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-ran-attn-direct` | `ran` | `attn` | https://doi.org/10.1016/j.neunet.2017.07.008 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-ran-proj-direct` | `ran` | `proj` | https://doi.org/10.1016/j.neunet.2017.07.008 | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-scrn-direct` | `scrn` | `direct` | https://doi.org/10.1007/978-3-319-11179-7_1 | - |
| `torch-rnnzoo-scrn-bidir-direct` | `scrn` | `bidir` | https://doi.org/10.1007/978-3-319-11179-7_1 | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-scrn-ln-direct` | `scrn` | `ln` | https://doi.org/10.1007/978-3-319-11179-7_1 | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-scrn-attn-direct` | `scrn` | `attn` | https://doi.org/10.1007/978-3-319-11179-7_1 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-scrn-proj-direct` | `scrn` | `proj` | https://doi.org/10.1007/978-3-319-11179-7_1 | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-rhn-direct` | `rhn` | `direct` | https://doi.org/10.21437/interspeech.2017-429 | - |
| `torch-rnnzoo-rhn-bidir-direct` | `rhn` | `bidir` | https://doi.org/10.21437/interspeech.2017-429 | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-rhn-ln-direct` | `rhn` | `ln` | https://doi.org/10.21437/interspeech.2017-429 | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-rhn-attn-direct` | `rhn` | `attn` | https://doi.org/10.21437/interspeech.2017-429 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-rhn-proj-direct` | `rhn` | `proj` | https://doi.org/10.21437/interspeech.2017-429 | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-clockwork-direct` | `clockwork` | `direct` | https://doi.org/10.1145/2576768.2598358 | - |
| `torch-rnnzoo-clockwork-bidir-direct` | `clockwork` | `bidir` | https://doi.org/10.1145/2576768.2598358 | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-clockwork-ln-direct` | `clockwork` | `ln` | https://doi.org/10.1145/2576768.2598358 | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-clockwork-attn-direct` | `clockwork` | `attn` | https://doi.org/10.1145/2576768.2598358 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-clockwork-proj-direct` | `clockwork` | `proj` | https://doi.org/10.1145/2576768.2598358 | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-qrnn-direct` | `qrnn` | `direct` | https://doi.org/10.1016/j.neunet.2016.04.001 | - |
| `torch-rnnzoo-qrnn-bidir-direct` | `qrnn` | `bidir` | https://doi.org/10.1016/j.neunet.2016.04.001 | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-qrnn-ln-direct` | `qrnn` | `ln` | https://doi.org/10.1016/j.neunet.2016.04.001 | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-qrnn-attn-direct` | `qrnn` | `attn` | https://doi.org/10.1016/j.neunet.2016.04.001 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-qrnn-proj-direct` | `qrnn` | `proj` | https://doi.org/10.1016/j.neunet.2016.04.001 | https://doi.org/10.21437/interspeech.2014-80 |
| `torch-rnnzoo-phased-lstm-direct` | `phased-lstm` | `direct` | https://doi.org/10.1109/ebccsp.2016.7605258 | - |
| `torch-rnnzoo-phased-lstm-bidir-direct` | `phased-lstm` | `bidir` | https://doi.org/10.1109/ebccsp.2016.7605258 | https://doi.org/10.1109/78.650093 |
| `torch-rnnzoo-phased-lstm-ln-direct` | `phased-lstm` | `ln` | https://doi.org/10.1109/ebccsp.2016.7605258 | https://arxiv.org/abs/1607.06450 |
| `torch-rnnzoo-phased-lstm-attn-direct` | `phased-lstm` | `attn` | https://doi.org/10.1109/ebccsp.2016.7605258 | https://doi.org/10.7575/aiac.alls.v.6n.4p.226 |
| `torch-rnnzoo-phased-lstm-proj-direct` | `phased-lstm` | `proj` | https://doi.org/10.1109/ebccsp.2016.7605258 | https://doi.org/10.21437/interspeech.2014-80 |
