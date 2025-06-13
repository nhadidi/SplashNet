# SplashNet Code

The code required to train and evalute our models is based on that released with the emg2qwerty dataset. The easiest way to use our models is to first set up an environment with the emg2qwerty code and dataset, and then to replace a few folders with those provided here.
## Setup

The below code block is copied from the README of the emg2qwerty repo. Follow these steps to set up the environment and download the dataset.

```shell
# Install [git-lfs](https://git-lfs.github.com/) (for pretrained checkpoints)
git lfs install

# Clone the repo, setup environment, and install local package
git clone git@github.com:facebookresearch/emg2qwerty.git ~/emg2qwerty
cd ~/emg2qwerty
conda env create -f environment.yml
conda activate emg2qwerty
pip install -e .

# Download the dataset, extract, and symlink to ~/emg2qwerty/data
cd ~ && wget https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz
tar -xvzf emg2qwerty-data-2021-08.tar.gz
ln -s ~/emg2qwerty-data-2021-08 ~/emg2qwerty/data
```

## emg2qwerty repository  modifications to run our models.
Once emg2qwerty setup is complete, simply replace the config and emg2qwerty folders in the cloned repository with the corresponding folders provided here. Then, copy splashnet.ckpt and splashnet-mini.ckpt into the models folder.


## 🛠 Model & Training Configuration Guide

> **Config files**  
> • Transforms `config/transforms/log_spectrogram.yaml`  
> • Model & Datamodule `config/model/tds_conv_ctc.yaml`

All configs have been initially set to reproduce the model of Sivakumar et al. 2024.
 
Our methods can be implemented through the following edits to the config files.

---

### 0. Longer Training Clips (16 s vs 4 s)

**`config/model/tds_conv_ctc.yaml`**
~~~yaml
datamodule:
  window_length: 32000   # 16 s, was 8000 (4 s)
~~~

---

### 1. Reduced Spectral Granularity (RSG – 6-bin spectrograms per electrode)

**`config/transforms/log_spectrogram.yaml`**
~~~yaml
logspec:
  _target_: emg2qwerty.transforms.NewLogSpectrogram   # was LogSpectrogram
  n_fft: 64        # unchanged
  hop_length: 16   # unchanged
~~~

**`config/model/tds_conv_ctc.yaml`**
~~~yaml
module:
  in_features: 96   # was 528
~~~

---

### 2. Rolling Time Normalization (RTN)


**`config/model/tds_conv_ctc.yaml`**
~~~yaml
module:
  spec_norm: 'RollingTimeNorm' # was 'BatchNorm2d'
~~~


### 3. Aggressive Channel Masking (ACM)

**`config/transforms/log_spectrogram.yaml`**
~~~yaml
specaug:
  _target_: emg2qwerty.transforms.SpecAugment
  n_time_masks: 0 # was 3
  time_mask_param: 25  # unchanged, doesn't matter since n_time_masks is 0
  n_freq_masks: 2 # unchanged
  freq_mask_param: 12 # was 4

~~~

---

### 4. Architectural Variants  
_All edits below go in **`config/model/tds_conv_ctc.yaml`**._

#### 4.1 Baseline (Joint-Hand)
~~~yaml
module:
  _target_: emg2qwerty.lightning.TDSConvCTCModule
~~~

#### 4.2 Split-and-Share (SplashNet-mini)
~~~yaml
_target_: emg2qwerty.lightning.TDSConvCTCModule
module:
  share_hand_weights: True
~~~

#### 4.3 Split-Only (or unshared finetuning for a pretrained Split-and-Share model, currently only compatible with RTN)
~~~yaml
_target_: emg2qwerty.lightning.TDSConvCTCFinetuneModule # was TDSConvCTCModule
module:
  share_hand_weights: True # Initializes with a shared-weight module before duplicating and unsharing weights
~~~


#### 4.4 Upscaled Split-and-Share (SplashNet)
~~~yaml
_target_: emg2qwerty.lightning.TDSConvCTCModule
module:
  share_hand_weights: True
  mlp_features: [528]        # was [384]
block_channels: [24, 24, 48, 48]   # was [24, 24, 24, 24]
~~~

---

## Training

To train models, we use the same commands as in the original emg2qwerty repository. Note that the sessions used for validation in our generic.yaml user config differ from those specified in the original emg2qwerty paper, as described in section 4.5 of our paper.

Generic user model:

```shell
python -m emg2qwerty.train \
  user=generic \
  trainer.accelerator=gpu trainer.devices=8 \
  --multirun
```

Personalized user models (replace the checkpoint argument with your desired checkpoint; we provide splashnet.ckpt and splashnet-mini.ckpt ):

```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  trainer.accelerator=gpu trainer.devices=1 \
  checkpoint="${HOME}/emg2qwerty/models/splashnet.ckpt" \
  --multirun
```

If you are using a Slurm cluster, include "cluster=slurm" override in the argument list of above commands to pick up `config/cluster/slurm.yaml`. This overrides the Hydra Launcher to use [Submitit plugin](https://hydra.cc/docs/plugins/submitit_launcher). Refer to Hydra documentation for the list of available launcher plugins if you are not using a Slurm cluster.

## Testing

Greedy decoding:

```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  checkpoint="${HOME}/emg2qwerty/models/splashnet.ckpt"
  train=False trainer.accelerator=gpu \
  decoder=ctc_greedy \
  hydra.launcher.mem_gb=64 \
  --multirun
```

Beam-search decoding with 6-gram character-level language model:

```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  checkpoint="${HOME}/emg2qwerty/models/splashnet.ckpt"
  train=False trainer.accelerator=gpu \
  decoder=ctc_beam \
  hydra.launcher.mem_gb=64 \
  --multirun
```

Though we do not provide the user-specific finetuned checkpoints in the attached supplementary material due to size constraints, a similar command to the above two (but specific to a given user and the corresponding finetuned model checkpoint) can be run to evaluate finetuned model perfomrance per user. For example, after finetuning a model on user0, run a command like:

```shell
python -m emg2qwerty.train \
  user="user0" \
  checkpoint=<user0_personalized_model_location>
  train=False trainer.accelerator=gpu \
  decoder=ctc_beam \
  hydra.launcher.mem_gb=64 \
  --multirun
```

In the paper, we report the average CER across participants for the four participants who were excluded in Sivakumar et al. 2024 as the "other domain validation set". To get performance on these participants, run the following command (shown here for beam search; just change decoder=ctc_beam to decoder=ctc_greedy to get the greedy decoding results). To make these evaluations easier to run without requiring much modification to the repository, we have two "user" yamls here: otheruser01.yaml and otheruser23.yaml. In each one, the score for one user's final session is reported as the "validation" CER, and that for the other user is reported as the "test" CER. For example, otheruser01.yaml has the final session for otheruser0 as its validation session, and the final session for otheruser1 as its test session. We ultimately average these metrics across the four participants to report the other user validation performance in the paper.

```shell
python -m emg2qwerty.train \
  user="glob(otheruser*)" \
  checkpoint="${HOME}/emg2qwerty/models/splashnet.ckpt"
  train=False trainer.accelerator=gpu \
  decoder=ctc_beam \
  hydra.launcher.mem_gb=64 \
  --multirun
```




## License

emg2qwerty is CC-BY-NC-4.0 licensed, as found in the LICENSE file.

## Citing emg2qwerty

```
@misc{sivakumar2024emg2qwertylargedatasetbaselines,
      title={emg2qwerty: A Large Dataset with Baselines for Touch Typing using Surface Electromyography},
      author={Viswanath Sivakumar and Jeffrey Seely and Alan Du and Sean R Bittner and Adam Berenzweig and Anuoluwapo Bolarinwa and Alexandre Gramfort and Michael I Mandel},
      year={2024},
      eprint={2410.20081},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.20081},
}
```