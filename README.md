# AttentionTuner
**Learning Memory Mechanisms for Decision Making through Demonstrations**

# Installtion
Run the following commands:
```shell
conda env create -f environment.yml
conda activate tuner
pip install -r requirements.txt
```

# Training
To train a model, run the following command:

```shell
python memory_gym/train.py --config <config-file>
```

or 

```shell
python ltmb/train.py --config <config-file>
```

depending on the benchmark you want to train on.


To see the full list of options/configs, run 

```shell
python ltmb/train.py --help
```

Config files used for experiments in the paper are located in [./memory_gym/configs/](./memory_gym/configs) and [./ltmb/configs/](./ltmb/configs).

# Evaluation

To evaluate a model, run the following command:

```shell
python memory_gym/eval.py --config <config-file> --model <final-model-checkpoint>
```

To see the full list of options, run:

```shell
python memory_gym/eval.py --help
```

# Datasets

See the LTMB and Memory Gym repositories for instructions on how to reproduce the datasets used in the paper.

# Citation
If you find **AttentionTuner** to be useful in your own research, please consider citing our paper:
