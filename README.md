# Maya: Multimodal Multilingual LLM

- Models and Dataset at [HuggingFace](https://huggingface.co/maya-multimodal)
- Paper: arXiv link TBD




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

### Pretraining

To pretrain the projection layer, 
- get the pretraining dataset from [HuggingFace](https://huggingface.co/maya-multimodal) and keep it in `/dev/data/LLaVA_Pretrain`
- get the images with `wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip` and keep them in `/dev/data/images`
  
```
bash scripts/maya/pretrain_aya_siglip.sh
```

### Instruction Tuning
Please download the annotations from [MBZUAI/palo_multilingual_dataset](https://huggingface.co/datasets/MBZUAI/palo_multilingual_dataset) and all images following the below links.


- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing),
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `/dev/data/instruction_tune_dataset/`,


```
instruction_tune_dataset
    ├── coco
    │   └── train2017
    ├── gqa
    │   └── images
    ├── ocr_vqa
    │   └── images
    ├── textvqa
    │   └── train_images
    └── vg
        ├── VG_100K
        └── VG_100K_2
```

Put the `palo_multilingual_dataset.json` in `/dev/data/annotations/palo_multilingual_dataset.json`

Make sure to keep the pretrained model you have in a path that you specify in the `scripts/maya/finetune_aya_siglip.sh` script throught the `--pretrain_mm_mlp_adapter` flag

Then run
```
bash scripts/maya/finetune_aya_siglip.sh
```

## Evaluation

For multilingual evaluation using PALO multilingual test dataset
- Download the PALO evaluation dataset: Create the following directory structure if it doesn't exist.
  ```
  LLaVA/playground/data/eval
  git clone https://huggingface.co/datasets/MBZUAI/multilingual-llava-bench-in-the-wild
  ```
- Run the evaluation script
```
bash scripts/v1_5/eval/eval_all_languages.sh \
    "model_base" \
    "model_path" \
    "model_name" \
    "your-openai-api-key"
```


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
- [Chen Liu](https://github.com/ccliu2)
- [Snegha A](https://github.com/Asnegha)
- [Anthony Susevski](https://github.com/asusevski)
- [Ashvanth.S](https://github.com/ash-01xor)
- [Genta Indra Winata](https://github.com/gentaiscool)
- [Ryan Chan](https://github.com/rchan26)
- [Sangyeon Kim](https://github.com/KimSangYeon-DGU)
- [Snehanshu](https://github.com/pilot-j)


## Acknowledgement

- This codebase is based on [LLaVA](https://github.com/haotian-liu/LLaVA). Thank you for the easily understable codebase.
- This project would not be possible without the support of Cohere and their Aya-35B API grant. We are thankful to Sara Hooker, Madeline, Shivalika, Shristhi and the entire Cohere for AI team for their support.
- We thank Pytho for their generaous GPU grant 



