# Data preparation

TODO: run `label_utterances.py`

***

# Installation

First, install the base conda environment and create a temporary directory where MFA 
will store the intermediate steps.

```bash
conda create -n mfa-dev -c conda-forge montreal-forced-aligner=2.2.17
conda activate mfa-dev  # activate the environment

export MFA_ROOT_DIR=/path/to/MFA/temp/directory  # set the root directory
mkdir ${MFA_ROOT_DIR}
```

Enable multiprocessing for MFA for quicker processes:

```bash
mfa configure --enable_mp --always_clean
```

Download the pretrained acoustic model, pronunciation dictionary and G2P model:

```bash
mfa model download acoustic english_mfa
mfa model download dictionary english_uk_mfa
mfa model download g2p english_uk_mfa
```

Generate pronunciations for the OOV words and update the pretrained
pronunciation dictionary with them:

```bash
export MOCHA_DIR=/path/to/MOCHA/TIMIT
mfa g2p ${MOCHA_DIR} english_uk_mfa ${MFA_ROOT_DIR}/g2pped_oovs.txt --dictionary_path english_uk_mfa
mfa model add_words english_uk_mfa ${MFA_ROOT_DIR}/g2pped_oovs.txt
```

Align the corpus:

```bash
export TMP_DIR=/path/where/to/save/the/alignments
mfa align ${MOCHA_DIR} english_uk_mfa english_mfa ${TMP_DIR}
rsync -a ${TMP_DIR} ${MOCHA_DIR}  # copy the alignments to the MOCHA-TIMIT directory
```

Now, the MOCHA-TIMIT dataset is in `${MOCHA_DIR}`.

# Sanity checks

Check the phone durations

```bash
python -m artprob.phone_durations \
--dataset=${MOCHA_DIR} \
--features=/path/to/artprob/phono_features/en_ipa.npz
```
