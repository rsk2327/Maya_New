# Maya: Multimodal Multilingual LLM

- Models and Dataset at [HuggingFace](https://huggingface.co/maya-multimodal)
- Paper: 




## Contents
- [Install](#install)
- [Model Weights and Dataset](#model-weights-and-dataset)
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

## Model Weights and Dataset
[HuggingFace](https://huggingface.co/maya-multimodal)


## Train

## Evaluation


## Citation

If you find Maya useful for your research and applications, please cite using this BibTeX:

## Contributors
In no particular order
- Team Leads: Nahid, Karthik, Surya
- [Timothy Chung](https://github.com/timothycdc)
- [Abhipsha Das](https://github.com/chiral-carbon)
- [Bala Krishna S Vegesna](https://github.com/Satyajitv)
- [Iftekhar Uddin](https://github.com/iuddin)
- [Drishti Sushma](https://github.com/DrishtiShrrrma)
- [Roshan Santhosh](https://github.com/rsk2327)
- [Shayakh Islam](https://github.com/shayekhbinislam)
- [Isha Chaturvedi](https://github.com/ishacusp)
- [Chen Liu]()
- [Snegha A]()
- [Anthony Susevski](https://github.com/asusevski)
- [Ashvanth.S]()
- [Genta Indra Winata]()
- [Ryan Chan](https://github.com/rchan26)
- [Sangyeon Kim](https://github.com/KimSangYeon-DGU)
- [Snehanshu](https://github.com/pilot-j)


## Acknowledgement

- This codebase is based on [LLaVA](https://github.com/haotian-liu/LLaVA). Thank you for the easily understable codebase.
- This project would not be possible without the support of Cohere and their Aya-35B API grant. We are thankful to Sara Hooker, Madeline, Shivalika, Shristhi and the entire Cohere for AI team for their support.
- We thank Pytho for their generaous GPU grant 



