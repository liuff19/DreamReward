# DreamReward: Text-to-3D Generation with Human Preference
[**Paper**]() | [**Project Page**](https://jamesyjl.github.io/DreamReward/) 
<p align="left">
    <a href="https://jamesyjl.github.io/">Junliang Ye</a><sup>*</sup></a><sup>1,</sup></a><sup>2</sup>&emsp;
    <a href="https://liuff19.github.io/">Fangfu Liu</a><sup>*1</sup>&emsp;
    Qixiu Li</a><sup>1</sup>&emsp;
    <a href="https://thuwzy.github.io/">Zhengyi Wang</a><sup>1,2</sup>&emsp;
    <a href="https://yikaiw.github.io/">Yikai Wang</a><sup>1</sup>&emsp;
    Xinzhou Wang</a><sup>1,2</sup>&emsp;
    <a href="https://duanyueqi.github.io/">Yueqi Duan</a><sup>1,&#x2709</sup>&emsp;
    <a href="https://ml.cs.tsinghua.edu.cn/~jun/index.shtml">Jun Zhu</a><sup>1,2,&#x2709</sup>&emsp;
</p>
<p align="left"><sup>1</sup>Tsinghua University &ensp; <sup>2</sup>ShengShu&ensp; <sup>*</sup> Equal Contribution<sup>&ensp; &#x2709</sup>  Corresponding Author</p>

<p align="center"> All Code will be released soon... 🏗️ 🚧 🔨</p>

Abstract: *3D content creation from text prompts has shown remarkable success recently. However, current text-to-3D methods often generate 3D results that do not align well with human preferences. In this paper, we present a comprehensive framework, coined DreamReward, to learn and improve text-to-3D models from human preference feedback. To begin with, we collect 25k expert comparisons based on a systematic annotation pipeline including rating and ranking. Then, we build Reward3D---the first general-purpose text-to-3D human preference reward model to effectively encode human preferences. Building upon the 3D reward model, we finally perform theoretical analysis and present the Reward3D Feedback Learning (DreamFL), a direct tuning algorithm to optimize the multi-view diffusion models with a redefined scorer. Grounded by theoretical proof and extensive experiment comparisons, our DreamReward successfully generates high-fidelity and 3D consistent results with significant boosts in prompt alignment with human intention. Our results demonstrate the great potential for learning from human feedback to improve text-to-3D models.*

<p align="center">
    <img src="assets/pipeline.jpg">
</p>

## Visual Results Compare with MVDream
<p align="center">
    <img src="assets/result7.png">
</p>
<p align="center">
    <img src="assets/new1-2.png">
</p>
<p align="center">
    <img src="assets/new2-2.png">
</p>


