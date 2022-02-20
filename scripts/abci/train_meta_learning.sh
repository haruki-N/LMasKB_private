#! /bin/sh
# run this script like 'qsub -cwd -g gcb50246 -l rt_G.small=1 -l h_rt=168:00:00 scripts/abci/train_meta_learning.sh'

### 以下、ABCIで必要な処理 ###
PYENV_NAME="miniconda3-4.0.5/envs/bart"
PATH=$HOME/.linuxbrew/bin:$HOME/.linuxbrew/sbin:$PATH

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate ${PYENV_NAME}  # 仮想環境の有効化

source /etc/profile.d/modules.sh
module load cuda/10.2/10.2.89  # load cuda
module load gcc/7.4.0  # load gcc
### ここまで ###

export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

DATE=`date +%Y%m%d-%H%M`
echo $DATE

hostname
uname -a
which python
python --version

python src/main_meta_learning.py ----children 180 --epoch_num 30 --batch_size 64 --lr 1e-5 --mix_ratio 10 --gpu_number 0
