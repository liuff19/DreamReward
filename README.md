<p align="center">
  <h3 align="center"><strong>DreamReward: Text-to-3D Generation with Human Preference</strong></h3>

<p align="center">
    <a href="https://jamesyjl.github.io/">Junliang Ye</a><sup>1,2*</sup>,
    <a href="https://liuff19.github.io/">Fangfu Liu</a><sup>1*</sup>,
    <a href="">Qixiu Li</a><sup>1</sup>,
    <a href="https://thuwzy.github.io/">Zhengyi Wang</a><sup>1,2</sup>,<br>
    <a href="https://yikaiw.github.io/">Yikai Wang</a><sup>1</sup>,
    <a href="https://zz7379.github.io/">Xinzhou Wang</a><sup>1,2</sup>,
    <a href="https://duanyueqi.github.io/">Yueqi Duan</a><sup>1†</sup>,
    <a href="https://ml.cs.tsinghua.edu.cn/~jun/index.shtml">Jun Zhu</a><sup>1,2†</sup>,
    <br>
    <sup>*</sup>Equal Contribution.
    <br>
    <sup>†</sup>Corresponding authors.
    <br>
    <sup>1</sup>Tsinghua University,
    <sup>2</sup>ShengShu,
</p>


<div align="center">

<a href='https://arxiv.org/abs/2403.14613'><img src='https://img.shields.io/badge/arXiv-2403.14613-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://jamesyjl.github.io/DreamReward/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/yejunliang23/Reward3D"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/datasets/yejunliang23/3DRewardDB"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HF-orange"></a>

</div>

<p align="center">
    <img src="asserts/pipeline_00.png" alt="pipeline method"/>
</p>

## Release
- [2025/3/19] 🔥🔥We released the DreamFL code of **DreamReward**.
- [2025/3/19] 🔥🔥We open-source two versions of **Reward3D** weights.

## Contents
- [Release](#release)
- [Contents](#contents)
- [Ablation](#Ablation)
- [Installation](#Installation)
- [Usage](#Usage)
- [Todo](#Todo)
- [Acknowledgement](#acknowledgement)
- [BibTeX](#bibtex)

## Ablation
We open-source two versions of Reward3D: **Reward3D_Scorer**, corresponding to the implementation presented in our paper, and **Reward3D_CrossViewFusion**, an extended version leveraging self-attention to enhance multi-view feature integration. **Furthermore**, we release two hyperparameter configurations of the DreamFL algorithm: DreamReward_guidance1, offering more conservative guidance, and DreamReward_guidance2, providing more aggressive guidance. Additionally, we provide ablation for further analysis and reference.
<p align="center">
    <img src="asserts/ablation_00.png">
</p>

## Installation
Our environment has been tested on Ubuntu 22, CUDA 11.8 with A100, A800.
1. Clone our repo and create conda environment
```
git clone https://github.com/liuff19/DreamReward.git && cd DreamReward
conda create -n dreamreward python=3.9
conda activate dreamreward
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.23.post1
pip install ninja
git clone https://github.com/bytedance/MVDream extern/MVDream
pip install -e extern/MVDream 
pip install -r requirements.txt
pip install fairscale
```
2. Install the pretrained Reward3D weight
```
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download yejunliang23/Reward3D --local-dir checkpoints/
```

## Usage
### Command line inference
```
# inference for CrossViewFusion1
python launch.py --config "configs/dreamreward1.yaml" --train --gpu 0 system.guidance.alg_type="Reward3D_CrossViewFusion" system.guidance_type="DreamReward-guidance1" name="CrossViewFusion1" system.prompt_processor.prompt="A delicate porcelain teacup, painted with intricate flowers, rests on a saucer" system.guidance.reward_model_path="checkpoints/Reward3D_CrossViewFusion.pt"
# or you can
bash scripts/cvf1.sh

# inference for CrossViewFusion2
python launch.py --config "configs/dreamreward2.yaml" --train --gpu 0 system.guidance.alg_type="Reward3D_CrossViewFusion" system.guidance_type="DreamReward-guidance1" name="CrossViewFusion2" system.prompt_processor.prompt="A delicate porcelain teacup, painted with intricate flowers, rests on a saucer" system.guidance.reward_model_path="checkpoints/Reward3D_CrossViewFusion.pt"
# or you can
bash scripts/cvf2.sh

# inference for Scorer1
python launch.py --config "configs/dreamreward1.yaml" --train --gpu 0 system.guidance.alg_type="Reward3D_Scorer" system.guidance_type="DreamReward-guidance1" name="Scorer1" system.prompt_processor.prompt="A delicate porcelain teacup, painted with intricate flowers, rests on a saucer" system.guidance.reward_model_path="checkpoints/Reward3D_Scorer.pt"
# or you can
bash scripts/scorer1.sh

# inference for Scorer2
python launch.py --config "configs/dreamreward2.yaml" --train --gpu 0 system.guidance.alg_type="Reward3D_Scorer" system.guidance_type="DreamReward-guidance1" name="Scorer2" system.prompt_processor.prompt="A delicate porcelain teacup, painted with intricate flowers, rests on a saucer" system.guidance.reward_model_path="checkpoints/Reward3D_Scorer.pt"
# or you can
bash scripts/scorer2.sh
```
## Todo
- [ ] Release the 3DReward-training code.

## Acknowledgement
Our code is based on these wonderful repos:
* **[ImageReward](https://github.com/THUDM/ImageReward)**
* **[MVDream](https://github.com/bytedance/MVDream)**
* **[threestudio](https://github.com/threestudio-project/threestudio)**

Also we invite you to explore our latest work [**DeepMesh**](https://zhaorw02.github.io/DeepMesh/) about applying *direct preference optimization* to 3D-mesh generation.

## BibTeX
```bibtex
@inproceedings{ye2024dreamreward,
  title={Dreamreward: Text-to-3d generation with human preference},
  author={Ye, Junliang and Liu, Fangfu and Li, Qixiu and Wang, Zhengyi and Wang, Yikai and Wang, Xinzhou and Duan, Yueqi and Zhu, Jun},
  booktitle={European Conference on Computer Vision},
  pages={259--276},
  year={2024},
  organization={Springer}
}
```
