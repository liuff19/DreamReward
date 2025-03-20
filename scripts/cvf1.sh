ckpt_reward="checkpoints/Reward3D_CrossViewFusion.pt"
gpu=3
name="CrossViewFusion1"
configs_neme="configs/dreamreward1.yaml"
alg_type="Reward3D_CrossViewFusion"
guidance_type="DreamReward-guidance1"
#prompt1:Phoenix
prompt="An ultra-detailed illustration of a mythical Phoenix, rising from ashes, vibrant feathers in a fiery palette."
python launch.py --config "$configs_neme" --train --gpu "$gpu" system.guidance.alg_type="$alg_type" system.guidance_type="$guidance_type" name="$name" system.prompt_processor.prompt="$prompt" system.guidance.reward_model_path="$ckpt_reward"
#prompt2:flowers teacup
prompt="A delicate porcelain teacup, painted with intricate flowers, rests on a saucer"
python launch.py --config "$configs_neme" --train --gpu "$gpu" system.guidance.alg_type="$alg_type" system.guidance_type="$guidance_type" name="$name" system.prompt_processor.prompt="$prompt" system.guidance.reward_model_path="$ckpt_reward"
#prompt3:bicycle
prompt="A bicycle that leaves a trail of flowers"
python launch.py --config "$configs_neme" --train --gpu "$gpu" system.guidance.alg_type="$alg_type" system.guidance_type="$guidance_type" name="$name" system.prompt_processor.prompt="$prompt" system.guidance.reward_model_path="$ckpt_reward"
#prompt4:fountain
prompt="A solid, symmetrical, smooth stone fountain, with water cascading over its edges into a clear, circular pond surrounded by blooming lilies, in the center of a sunlit courtyard"
python launch.py --config "$configs_neme" --train --gpu "$gpu" system.guidance.alg_type="$alg_type" system.guidance_type="$guidance_type" name="$name" system.prompt_processor.prompt="$prompt" system.guidance.reward_model_path="$ckpt_reward"
#prompt5:pen
prompt="A pen leaking blue ink"
python launch.py --config "$configs_neme" --train --gpu "$gpu" system.guidance.alg_type="$alg_type" system.guidance_type="$guidance_type" name="$name" system.prompt_processor.prompt="$prompt" system.guidance.reward_model_path="$ckpt_reward"
#prompt6:mouse
prompt="A marble bust of a mouse"
python launch.py --config "$configs_neme" --train --gpu "$gpu" system.guidance.alg_type="$alg_type" system.guidance_type="$guidance_type" name="$name" system.prompt_processor.prompt="$prompt" system.guidance.reward_model_path="$ckpt_reward"
#prompt7:telephone
prompt="A rotary telephone carved out of wood"
python launch.py --config "$configs_neme" --train --gpu "$gpu" system.guidance.alg_type="$alg_type" system.guidance_type="$guidance_type" name="$name" system.prompt_processor.prompt="$prompt" system.guidance.reward_model_path="$ckpt_reward"
#prompt8:old hat
prompt="A torn hat"
python launch.py --config "$configs_neme" --train --gpu "$gpu" system.guidance.alg_type="$alg_type" system.guidance_type="$guidance_type" name="$name" system.prompt_processor.prompt="$prompt" system.guidance.reward_model_path="$ckpt_reward"
#prompt9:campfire
prompt="A smoldering campfire under a clear starry night, embers glowing softly"
python launch.py --config "$configs_neme" --train --gpu "$gpu" system.guidance.alg_type="$alg_type" system.guidance_type="$guidance_type" name="$name" system.prompt_processor.prompt="$prompt" system.guidance.reward_model_path="$ckpt_reward"
#prompt10:frog
prompt="Frog with a translucent skin displaying a mechanical heart beating."
python launch.py --config "$configs_neme" --train --gpu "$gpu" system.guidance.alg_type="$alg_type" system.guidance_type="$guidance_type" name="$name" system.prompt_processor.prompt="$prompt" system.guidance.reward_model_path="$ckpt_reward"
#prompt11:book
prompt="A book bound in mysterious symbols"
python launch.py --config "$configs_neme" --train --gpu "$gpu" system.guidance.alg_type="$alg_type" system.guidance_type="$guidance_type" name="$name" system.prompt_processor.prompt="$prompt" system.guidance.reward_model_path="$ckpt_reward"
#prompt12:pen and manuscripts
prompt="A pen sitting atop a pile of manuscripts"
python launch.py --config "$configs_neme" --train --gpu "$gpu" system.guidance.alg_type="$alg_type" system.guidance_type="$guidance_type" name="$name" system.prompt_processor.prompt="$prompt" system.guidance.reward_model_path="$ckpt_reward"
