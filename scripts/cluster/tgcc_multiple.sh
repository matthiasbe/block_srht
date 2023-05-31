#!/bin/sh

# XOR

BSRHT_RPLC=false \
BSRHT_SEP="','" \
BSRHT_LS=[600] \
BSRHT_MATRIX_FILE=../datasets/xor_pot16_129 \
BSRHT_SIGMA=1e3 \
BSRHT_NBEXP=1 \
BSRHT_KS="200:100:600" \
BSRHT_PRELOAD=false \
ccc_msub -N 2 -n 64 -E "--mem-per-cpu=5500" scripts/tgcc.sh

BSRHT_RPLC=false \
BSRHT_SEP="','" \
BSRHT_LS=[1000] \
BSRHT_MATRIX_FILE=../datasets/xor_pot16_129 \
BSRHT_SIGMA=1e3 \
BSRHT_NBEXP=1 \
BSRHT_KS="200:100:600" \
BSRHT_PRELOAD=false \
ccc_msub -N 2 -n 64 -E "--mem-per-cpu=5500" scripts/tgcc.sh

BSRHT_RPLC=false \
BSRHT_SEP="','" \
BSRHT_LS=[2000] \
BSRHT_MATRIX_FILE=../datasets/xor_pot16_129 \
BSRHT_SIGMA=1e3 \
BSRHT_NBEXP=1 \
BSRHT_KS="200:100:600" \
BSRHT_PRELOAD=false \
ccc_msub -N 2 -n 64 -E "--mem-per-cpu=5500" scripts/tgcc.sh

# YEAR

BSRHT_RPLC=false \
BSRHT_SEP="' '" \
BSRHT_LS=[2000] \
BSRHT_MATRIX_FILE=../datasets/year_pot16_90 \
BSRHT_SIGMA=1e4 \
BSRHT_NBEXP=1 \
BSRHT_KS="200:100:600" \
BSRHT_PRELOAD=false \
ccc_msub -N 2 -n 64 -E "--mem-per-cpu=5500" scripts/tgcc.sh

BSRHT_RPLC=false \
BSRHT_SEP="' '" \
BSRHT_LS=[600] \
BSRHT_MATRIX_FILE=../datasets/year_pot16_90 \
BSRHT_SIGMA=1e4 \
BSRHT_NBEXP=1 \
BSRHT_KS="200:100:600" \
BSRHT_PRELOAD=false \
ccc_msub -N 2 -n 64 -E "--mem-per-cpu=5500" scripts/tgcc.sh

# BOTNET

BSRHT_RPLC=false \
BSRHT_SEP="','" \
BSRHT_LS=[2000] \
BSRHT_MATRIX_FILE=../datasets/botnet_pot16_115 \
BSRHT_SIGMA=1e3 \
BSRHT_NBEXP=1 \
BSRHT_KS="200:100:600" \
BSRHT_PRELOAD=false \
ccc_msub -N 2 -n 64 -E "--mem-per-cpu=5500" scripts/tgcc.sh

BSRHT_RPLC=false \
BSRHT_SEP="','" \
BSRHT_LS=[1000] \
BSRHT_MATRIX_FILE=../datasets/botnet_pot16_115 \
BSRHT_SIGMA=1e3 \
BSRHT_NBEXP=1 \
BSRHT_KS="200:100:600" \
BSRHT_PRELOAD=false \
ccc_msub -N 2 -n 64 -E "--mem-per-cpu=5500" scripts/tgcc.sh

BSRHT_RPLC=false \
BSRHT_SEP="','" \
BSRHT_LS=[600] \
BSRHT_MATRIX_FILE=../datasets/botnet_pot16_115 \
BSRHT_SIGMA=1e3 \
BSRHT_NBEXP=1 \
BSRHT_KS="200:100:600" \
BSRHT_PRELOAD=false \
ccc_msub -N 2 -n 64 -E "--mem-per-cpu=5500" scripts/tgcc.sh

# EPSILON

BSRHT_RPLC=false \
BSRHT_SEP="' '" \
BSRHT_LS=[2000] \
BSRHT_MATRIX_FILE=../datasets/epsilon_pot16_2000 \
BSRHT_SIGMA=1e2 \
BSRHT_NBEXP=1 \
BSRHT_KS="200:100:600" \
BSRHT_PRELOAD=false \
ccc_msub -N 2 -n 64 -E "--mem-per-cpu=5500" scripts/tgcc.sh

BSRHT_RPLC=false \
BSRHT_SEP="' '" \
BSRHT_LS=[1000] \
BSRHT_MATRIX_FILE=../datasets/epsilon_pot16_2000 \
BSRHT_SIGMA=1e2 \
BSRHT_NBEXP=1 \
BSRHT_KS="200:100:600" \
BSRHT_PRELOAD=false \
ccc_msub -N 2 -n 64 -E "--mem-per-cpu=5500" scripts/tgcc.sh

BSRHT_RPLC=false \
BSRHT_SEP="' '" \
BSRHT_LS=[600] \
BSRHT_MATRIX_FILE=../datasets/epsilon_pot16_2000 \
BSRHT_SIGMA=1e2 \
BSRHT_NBEXP=1 \
BSRHT_KS="200:100:600" \
BSRHT_PRELOAD=false \
ccc_msub -N 2 -n 64 -E "--mem-per-cpu=5500" scripts/tgcc.sh

