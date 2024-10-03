#!/bin/bash
set -e

# set constants
LOCAL_HOME=     # ¡complete this!
DATASETS=       # ¡complete this!
MFA=            # ¡complete this!: temporary directory for MFA's files

# comment out this line after setting the constants above
exit 0

ORIG_MOCHA_TIMIT=${DATASETS}/mocha-timit-full   # original (downloaded) MOCHA-TIMIT dataset
DER_MOCHA_TIMIT=${DATASETS}/mocha-timit         # derived sub-dataset of the original with a tree structure following the README.md 
MOCHA_TIMIT=${DATASETS}/mocha-timit-clean       # the clean data will be written here

RED='\033[0;31m'
NC='\033[0m'

echo -e "${RED}Forced alignment: MOCHA-TIMIT${NC}"

# clone the unannotated MOCHA-TIMIT directory
cp -r ${DER_MOCHA_TIMIT} ${MOCHA_TIMIT}

# add the transcription to the MOCHA-TIMIT speakers
echo -e "${RED}Adding transcriptions for all waveform recordings...${NC}"
source /path/to/artprob/artprobenv/bin/activate     # ¡change the path to artprob!
python -m artprob.ema_eval.label_utterances \
       --path=${ORIG_MOCHA_TIMIT}/mocha-timit.txt \
       --mocha=${MOCHA_TIMIT}
deactivate
echo -e "${RED}...done!${NC}"

# optional: load conda 
# source /path/to/*conda[3]/etc/profile.d/conda.sh

# initialise the forced-alignment environment
echo -e "${RED}Initialising the forced-alignment environment...${NC}"
mkdir -p ${MFA}
conda create -n mfa-dev -c conda-forge montreal-forced-aligner=2.2.17
conda activate mfa-dev
export MFA_ROOT_DIR=${MFA}
mfa configure --enable_mp --always_clean
echo -e "${RED}...done!${NC}"

# download the pretrained acoustic model, pronunciation dictionary and G2P model
echo -e "${RED}Downloading the pretrained acoustic model, pronunciation dictionary and G2P model...${NC}"
mfa model download acoustic english_mfa
mfa model download dictionary english_uk_mfa
mfa model download g2p english_uk_mfa
echo -e "${RED}...done!${NC}"

# generate pronunciations for the OOV words and update the pretrained pronunciation dictionary with them
echo -e "${RED}Updating the pretrained pronunciation dictionary with those of OOV words'...${NC}"
mfa g2p ${MOCHA_TIMIT} english_uk_mfa ${MFA}/g2pped_oovs.txt --dictionary_path english_uk_mfa
mfa model add_words english_uk_mfa ${MFA}/g2pped_oovs.txt
echo -e "${RED}...done!${NC}"

# align the corpus
echo -e "${RED}Forced aligning the MOCHA-TIMIT corpus...${NC}"
mfa align ${MOCHA_TIMIT} english_uk_mfa english_mfa ${DATASETS}/mocha-timit-aligned
conda deactivate
echo -e "${RED}...done!${NC}"

# copy the phonetic alignment into the MOCHA-TIMIT directory
echo -e "${RED}Copying the phonetic alignment into the MOCHA-TIMIT directory...${NC}"
rsync -a ${DATASETS}/mocha-timit-aligned/ ${MOCHA_TIMIT}
echo -e "${RED}...done!${NC}"
