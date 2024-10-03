# Simulating articulatory trajectories with phonological feature interpolation

## Installation

First, clone this repository:

```bash
git clone https://github.com/angelo-ortiz/articulatory-probing.git
cd articulatory-probing
```

Before installing this package, it's best to create a virtual environment.
<details open>
<summary>If you have Python 3.10 installed in your OS and you don't want to use Anaconda (or its 
derivatives), you can use <code>venv</code> for that:</summary>

```bash
python3.10 -m venv ./artprobenv
source ./artprobenv/bin/activate
```

</details>
<details>
<summary>Otherwise, you can create a conda environment:</summary>

```bash
conda create -n artprobenv python=3.10
conda activate artprobenv
```

</details>

Finally, install the package within the virtual environment:

```bash
python3.10 -m pip install -e .[plot]  # add plot support for the notebooks
```

## Dataset processing
### Forced alignment
Run the forced alignment [script](forced_align.sh) after completing/replacing the lines
commented with `¡...!`.

### Generation of articulatory parameters from EMA traces
Follow the instructions in https://github.com/georgesma/agent. Notably, you need to
import LPCNet and preprocess the MOCHA-TIMIT dataset with `python preprocess_datasets.py`

> [!NOTE]
> You can also directly complete the MOCHA-TIMIT dataset with the forced alignments and
articulatory parameters in the complements [file](mocha_timit_complements.tar.zst).

## Dataset tree structure
After the pre-processing, the MOCHA-TIMIT dataset should be grouped by speaker 
(`fsew0`, `faet0`, `fsew0`, `maps0`, `mjjn0`, `msak0`), and each speaker directory 
should contain two sub-directories named `train` and `test`.
For each utterance, there should be at least 5 files: `.TextGrid` 
(MFA forced-alignment), `.ema` (EMA traces), `.param` (derived articulatory parameters),
`.trans` (transcription) and `.wav` (waveform).

In the end, the dataset tree should look like the structure below
```
MOCHA_TIMIT  
│
│─── fsew0
│    │─── train
│    │    │─── fsew0_001.TextGrid
│    │    │─── fsew0_001.ema
│    │    │─── fsew0_001.param
│    │    │─── fsew0_001.trans
│    │    │─── fsew0_001.wav
│    │    │─── fsew0_002.TextGrid
│    │    │─── fsew0_002.ema
│    │    │─── ...
│    │    └─── fsew0_410.wav
│    │
│    └─── test
│         │─── fsew0_411.TextGrid
│         │─── fsew0_411.ema
│         │─── fsew0_411.param
│         │─── fsew0_411.trans
│         │─── fsew0_411.wav
│         │─── fsew0_412.TextGrid
│         │─── fsew0_412.ema
│         │─── ...
│         └─── fsew0_460.wav
│         
│─── faet0
│    └─── ...
│─── ffes0
│    └─── ...
│─── maps0
│    └─── ...
│─── mjjn0
│    └─── ...
└─── msak0
     └─── ...
```

## Examples

1. To **train** a linear interpolation model on MOCHA-TIMIT's speaker `fsew0` with the one-hot 
Articulatory Phonology feature set, run
    ```bash
    MOCHA_TIMIT=/path/to/mocha/timit
    artprob train --batch_size 8 --seed 42 --dataset ${MOCHA_TIMIT}/fsew0 \
      --ema_sample_rate 100 --art_params --remove_silence --model linear \
      --features ./phono_features/en_ap_1hot.npz --language us \
      --keep_unknown_phon_feats --ignore_phnm_cache --boundary_knots mid probing \
      --type linear --epoch 100 --checkpoint_path=/path/to/checkpoints/linear \
      --save_step 5
    ```

2. To **train** a cubic Hermite interpolation model with temporal and spatial optimisation on 
MOCHA-TIMIT's speaker `maps0` with the Generative Phonology feature set enriched with
one-hot phonemes, run
    ```bash
    artprob train --batch_size 8 --seed 42 --dataset ${MOCHA_TIMIT}/maps0 \
      --ema_sample_rate 100 --art_params --remove_silence --model ma-cubic \
      --features ./phono_features/en_ipa_phnm.npz --language uk \
      --keep_unknown_phon_feats --ignore_phnm_cache --max_iter 10 --lr 1e-6 \
      --knot_lr 1e-2 --weak_coeff 1e4 probing --type linear --epoch 100 \
      --checkpoint_path=/path/to/checkpoints/ma-cubic --save_step 5
    ```

3. To **test** the cubic Hermite interpolation above, run
    ```bash
    artprob test --dataset ${MOCHA_TIMIT}/maps0 --model ma-cubic \
      --results_path /path/to/results/ma-cubic \
      --checkpoint_path /path/to/checkpoints/ma-cubic
    ```
