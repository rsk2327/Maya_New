# Maya: Multimodal Multilingual LLM

- Models and Dataset at [HuggingFace](https://huggingface.co/maya-multimodal)
- Paper: 




## Contents
- [Install](#install)
- [Model Weights](#model-weights)
- [Demo](#Demo)
- [Dataset](#dataset)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

1. Clone this repository and navigate to maya directory
```bash
git clone https://github.com/nahidalam/maya
cd maya
```

2. Install Package
```Shell
conda create -n maya python=3.10 -y
conda activate maya
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

4. We upgraded a few libraries so install them now or feel free to update the `pyproject.toml` file
```
pip install chardet==5.2.0 datasets==2.15.0 deepspeed==0.14.2 fastapi==0.111.0 transformers==4.42.3 accelerate==0.27.2

```

## Model Weights

## Demo

## Dataset

## Train

## Evaluation


## Citation

If you find Maya useful for your research and applications, please cite using this BibTeX:

## Contributors
- Team Leads: Nahid, Karthik, Surya
- Satya https://github.com/Satyajitv
- Iftekhar Uddin https://github.com/iuddin
- Drishti Sushma  https://github.com/DrishtiShrrrma
- Roshan Santhosh https://github.com/rsk2327
- Cecilia Liu, Snegha, Shayakh, Anthony, Isha
- Ryan Chan https://github.com/rchan26
- Sangyeon Kim https://github.com/KimSangYeon-DGU
- Snehanshu https://github.com/pilot-j


## Acknowledgement

- This project would not be possible without the support of Cohere and their Aya-35B API grant. We are thankful to Sara Hooker, Madeline, Shivalika, Shristhi and the entire Cohere for AI team for
- We thank Pytho for their generaous GPU grant 
- This codebase is based on [LLaVA](https://github.com/haotian-liu/LLaVA) Thank you for the easily understable codebase. 



