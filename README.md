# This is an implementation of a novel game theoretic method for imitation learning. 

# Reproducing coinrun expriements in Colab: Recommended way. 

```
import os
del os.environ['LD_PRELOAD']
!apt-get remove libtcmalloc*

!apt-get update
!apt-get install mpich build-essential qt5-default pkg-config

!git clone https://github.com/openai/coinrun

!pip install -r coinrun/requirements.txt

# alter sys path instead of pip install -e to avoid colab import issues
import sys
sys.path.insert(0, 'coinrun')

from google.colab import drive
drive.mount('/content/gdrive')

%cp -r gdrive/My\ Drive/codes/ .
%cd codes/
%cd corgail/
%cp -r /content/coinrun/coinrun .
!unzip gail_experts.zip

!git clone https://github.com/openai/baselines.git
%cd baselines
!pip install -e .
%cd ..

```

# CorGAIL: CoinRun environment
```
!python main.py --env-name CoinRun --num-levels 0 --high-difficulty False --algo ppo --cor-gail --use-gae --log-interval 10 --num-processes 8  --lr 3e-5 --entropy-coef 0.02 --num-steps 32 --ppo-epoch 10 --gail-epoch 1 --queue-size 5 --embed-size 2 --gamma 0.99 --gae-lambda 0.95  --num-env-steps 2000000 --use-linear-lr-decay --use-proper-time-limits --seed 1 --save-interval 100 --save-dir ./trained_models/coinrun/
 
!python main.py --env-name CoinRun --num-levels 0 --high-difficulty False --algo ppo --cor-gail --use-gae --log-interval 10 --num-processes 8 --lr 3e-5 --entropy-coef 0.02 --num-steps 32 --ppo-epoch 10 --gail-epoch 1 --queue-size 5 --embed-size 2 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 2000000 --use-linear-lr-decay --use-proper-time-limits --seed 3 --save-interval 100 --save-dir ./trained_models/coinrun/
 
!python main.py --env-name CoinRun --num-levels 0 --high-difficulty False --algo ppo --cor-gail --use-gae --log-interval 10 --num-processes 8 --lr 3e-5 --entropy-coef 0.02 --num-steps 32 --ppo-epoch 10 --gail-epoch 1 --queue-size 5 --embed-size 2 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 2000000 --use-linear-lr-decay --use-proper-time-limits --seed 5 --save-interval 100 --save-dir ./trained_models/coinrun/ 
```

# CorGAIL: For classic environment. 
## For classic environments smaller pruning leads to faster learning. Please change inc from 2000 to 100 in utils.py -> ```queue_update(queue, m, K, t, ft, inc=100)```
```
!python main.py --env-name Pendulum-v0 --algo ppo --cor-gail --use-gae --log-interval 10 --num-steps 64 --num-processes 8 --lr 3e-5 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --gail-epoch 1 --num-mini-batch 8 --gamma 0.99 --gae-lambda 0.95 --queue-size 10 --num-env-steps 5000000 --seed 0 --log-dir logs/pendulum/corgail-0/

!python main.py --env-name Pendulum-v0 --algo ppo --cor-gail --use-gae --log-interval 10 --num-steps 64 --num-processes 8 --lr 3e-5 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --gail-epoch 1 --num-mini-batch 8 --gamma 0.99 --gae-lambda 0.95 --queue-size 10 --num-env-steps 5000000 --seed 1 --log-dir logs/pendulum/corgail-1/

!python main.py --env-name Pendulum-v0 --algo ppo --cor-gail --use-gae --log-interval 10 --num-steps 64 --num-processes 8 --lr 3e-5 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --gail-epoch 1 --num-mini-batch 8 --gamma 0.99 --gae-lambda 0.95 --queue-size 10 --num-env-steps 5000000 --seed 2 --log-dir logs/pendulum/corgail-2/ 
 ```

 # wGAIL: CoinRun environment
```
!python main.py --env-name CoinRun --num-levels 0 --high-difficulty False --algo ppo --gail --use-gae --log-interval 10 --num-processes 8 --lr 3e-5 --entropy-coef 0.02 --num-steps 32 --ppo-epoch 10 --gail-epoch 5  --gamma 0.99 --gae-lambda 0.95 --num-env-steps 2000000 --use-linear-lr-decay --use-proper-time-limits --seed 1 --save-interval 100 --save-dir ./trained_models/coinrun/
 
!python main.py --env-name CoinRun --num-levels 0 --high-difficulty False --algo ppo --gail --use-gae --log-interval 10 --num-processes 8 --lr 3e-5 --entropy-coef 0.02 --num-steps 32 --ppo-epoch 10 --gail-epoch 5  --gamma 0.99 --gae-lambda 0.95 --num-env-steps 2000000 --use-linear-lr-decay --use-proper-time-limits --seed 3 --save-interval 100 --save-dir ./trained_models/coinrun/

!python main.py --env-name CoinRun --num-levels 0 --high-difficulty False --algo ppo --gail --use-gae --log-interval 10 --num-processes 8 --lr 3e-5 --entropy-coef 0.02 --num-steps 32 --ppo-epoch 10 --gail-epoch 5  --gamma 0.99 --gae-lambda 0.95 --num-env-steps 2000000 --use-linear-lr-decay --use-proper-time-limits --seed 5 --save-interval 100 --save-dir ./trained_models/coinrun/
``` 
 
# To reproduce locally (has been only tested on Mac, python3).
## Install coinrun env from https://github.com/openai/coinrun

```
unzip gail_experts.zip
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..
```
 
# CorGAIL-Pendulum: for visualization, point the visualize_training_dynamics.py to the logs/pendulum/ .For classic environments smaller pruning leads to faster learning. Please change inc from 2000 to 100 in utils.py -> ```queue_update(queue, m, K, t, ft, inc=100)```
```
python main.py --env-name Pendulum-v0 --algo ppo --cor-gail --use-gae --log-interval 10 --num-steps 64 --num-processes 8 --lr 3e-5 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --gail-epoch 1 --num-mini-batch 8 --gamma 0.99 --gae-lambda 0.95 --queue-size 10 --num-env-steps 5000000 --seed 0 --log-dir logs/pendulum/corgail-0/

python main.py --env-name Pendulum-v0 --algo ppo --cor-gail --use-gae --log-interval 10 --num-steps 64 --num-processes 8 --lr 3e-5 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --gail-epoch 1 --num-mini-batch 8 --gamma 0.99 --gae-lambda 0.95 --queue-size 10 --num-env-steps 5000000 --seed 1 --log-dir logs/pendulum/corgail-1/

python main.py --env-name Pendulum-v0 --algo ppo --cor-gail --use-gae --log-interval 10 --num-steps 64 --num-processes 8 --lr 3e-5 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --gail-epoch 1 --num-mini-batch 8 --gamma 0.99 --gae-lambda 0.95 --queue-size 10 --num-env-steps 5000000 --seed 2 --log-dir logs/pendulum/corgail-2/
```

# CorGAIL coinrun:
## for visualization: Coinrun automatically saves the results in a directory. The directory name gets printed at the beginning. You have to put all the directories in any arbitrary directory with the results directory saved up with seeds-number at the end. For example `coinrun-1' `coinrun-3' `coinrun-5' in a parent directoy `coinrun'. Then point the visualize_training_dynamics.py to the logs/coinrun/

```
python main.py --env-name CoinRun --num-levels 0 --high-difficulty False --algo ppo --cor-gail --use-gae --log-interval 10 --num-processes 8 --lr 3e-5 --entropy-coef 0.02 --num-steps 32 --ppo-epoch 10 --gail-epoch 1 --queue-size 5 --embed-size 2 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 2000000 --use-linear-lr-decay --use-proper-time-limits --seed 1 --save-interval 100 --save-dir ./trained_models/coinrun/
 
python main.py --env-name CoinRun --num-levels 0 --high-difficulty False --algo ppo --cor-gail --use-gae --log-interval 10 --num-processes 8 --lr 3e-5 --entropy-coef 0.02 --num-steps 32 --ppo-epoch 10 --gail-epoch 1 --queue-size 5 --embed-size 2 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 2000000 --use-linear-lr-decay --use-proper-time-limits --seed 3 --save-interval 100 --save-dir ./trained_models/coinrun/
 
python main.py --env-name CoinRun --num-levels 0 --high-difficulty False --algo ppo --cor-gail --use-gae --log-interval 10 --num-processes 8  --lr 3e-5 --entropy-coef 0.02 --num-steps 32 --ppo-epoch 10 --gail-epoch 1 --queue-size 5 --embed-size 2 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 2000000 --use-linear-lr-decay --use-proper-time-limits --seed 5 --save-interval 100 --save-dir ./trained_models/coinrun/
```

# Acknowledgments.
The classical gail and ppo algorithms are from <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail>.

