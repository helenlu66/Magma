import os
import json
import torch
import numpy as np
from magma.image_processing_magma import MagmaImageProcessor
from magma.processing_magma import MagmaProcessor
from magma.modeling_magma import MagmaForConditionalGeneration
from data.openx.action_tokenizer import ActionTokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)

def get_magma_model(model_name):
    processor = MagmaProcessor.from_pretrained(model_name, trust_remote_code=True) 
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Magma-8B",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    return processor, model

def get_magma_prompt(task_description, processor, model_config):
    convs = [
    # {"role": "system", "content": "You are an agent that can see, talk and act."},
    # {"role": "user", "content": "<image>\nDo you see a red block?"},
    # {"role": "assistant", "content": "yes, there is a red block in the image."},
    # {"role": "user", "content": "Where is it?"},
    # {"role": "assistant", "content": "TThe red block is in the air, flying above a blue block."},
    # {"role": "user", "content": f"{task_description}"}
    # {"role": "user", "content": "What is it like to see a red cube?"},
    # {"role": "assistant", "content": "Seeing a red cube is visually striking, as it stands out against the background. The bright color and distinct shape of the cube can draw attention and create a sense of contrast with other objects in the scene."}
]
    convs = [
        {"role": "user", "content": f"<image>\n{task_description}?"},
    ]
    # convs = [
    #     {
    #         "role": "system",
    #         "content": "You are agent that can see, talk and act.", 
    #     },            
    # ] + convs      
    prompt = processor.tokenizer.apply_chat_template(
        convs,
        tokenize=False,
        add_generation_prompt=True
    )
    if model_config.mm_use_image_start_end:
        prompt = prompt.replace("<image>", "<image_start><image><image_end>")    
    return prompt

def get_magma_action(magma, processor, img, prompt, task_suite_name):
    # dataset_stats = json.load(open(os.path.join(magma.config._name_or_path, "dataset_statistics.json")))
    # action_norm_stats = dataset_stats[f"{task_suite_name}_no_noops"]['action']
    # n_action_bins = 256
    # vocab_size = processor.tokenizer.vocab_size
    # bins = np.linspace(-1, 1, n_action_bins)
    # bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # process inputs
    inputs = processor(images=img, texts=prompt, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)   
    inputs = inputs.to("cuda").to(torch.bfloat16)
    generation_args = {
        "max_new_tokens": 1000,
        "temperature": 0.7, 
        "do_sample": True,
        "num_beams": 1,
        "use_cache": False,  # Disabled to avoid DynamicCache compatibility issues
    }
    # predict actions with magma
    with torch.inference_mode():
        generate_ids = magma.generate(**inputs, **generation_args)

    action_ids = generate_ids[0, -8:-1].cpu().tolist()
    predicted_action_ids = np.array(action_ids).astype(np.int64)
    # discretized_actions = vocab_size - predicted_action_ids
    # discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=bin_centers.shape[0] - 1)
    # normalized_actions = bin_centers[discretized_actions]

    # # unnormalize actions
    # mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    # action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    # raw_action = np.where(
    #     mask,
    #     0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
    #     normalized_actions,
    # )
    # action = normalize_gripper_action(raw_action, binarize=True)
    # action = invert_gripper_action(action)

    # return action
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    normalized_action = action_tokenizer.decode_token_ids_to_actions(predicted_action_ids)
    return normalized_action

def unnormalize_action(normalized_action, action_stats):
    action_low, action_high = np.array(action_stats["q01"]), np.array(action_stats["q99"])
    return 0.5 * (normalized_action + 1) * (action_high - action_low) + action_low

def normalize_gripper_action(action, binarize=True):
    """
    Convert gripper action from [0,1] to [-1,+1] range.
    y = 2x - 1
    """
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action

def invert_gripper_action(action):
    """Convert gripper: RLDS(0=close,1=open) -> -1=open,+1=close"""
    action[..., -1] = action[..., -1] * -1.0
    return action