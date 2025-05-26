# VeriFree: Reinforcing General Reasoning without Verifiers

## Dependency

The code has been tested in the following environment: 

```
conda create -n VeriFree python=3.10 -y
conda activate VeriFree

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.8.4
pip install -U oat-llm==0.1.0

# PATH_TO_YOUR_USER_DIRECTORY in the following coammand should be modified 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{PATH_TO_YOUR_USER_DIRECTORY}/.conda/envs/VeriFree/lib/
```

## Training

The following command is an example for fine-tuning Qwen3 base models by VeriFree policy optimization:

```
bash run.sh
```
