import os
import numpy as np
import draccus
from dataclasses import dataclass
from typing import Optional, Tuple
import tqdm

from robosuite.controllers import load_controller_config
from robosuite.environments.base import MujocoEnv
from robosuite.wrappers import VisualizationWrapper
import robosuite as suite
from mimicgen_env_red_green import (
    get_libero_env, 
    get_libero_dummy_action,
    get_libero_obs,
    get_max_steps,
    set_seed_everywhere,
    save_rollout_video
)
from mimicgen_magma_red_green_utils import (
    get_magma_model,
    get_magma_prompt,
    get_magma_action
)

@dataclass
class MimicgenConfig:
    # Model parameters
    model_name: str = "microsoft/Magma-8b"      # model_name
    task_suite_name: str = "libero_goal"                    # Task suite name
    
    # Evaluation parameters
    num_trials_per_task: int = 50                          # Number of rollouts per task
    resolution: int = 256                                  # Image resolution
    num_steps_wait: int = 10                              # Steps to wait for stabilization
    seed: int = 0                                         # Random seed
    save_dir: str = "./mimicgen_eval_log"                   # Directory for saving logs

@draccus.wrap()
def eval_mimicgen(cfg: MimicgenConfig) -> Tuple[int, int]:
    """
    Evaluate Libero environment with given configuration.
    
    Args:
        cfg: LiberoConfig object containing evaluation parameters
        
    Returns:
        Tuple[int, int]: Total episodes and total successful episodes
    """
    

    
    # Initialize counters
    total_episodes, total_successes = 0, 0
    set_seed_everywhere(cfg.seed)
    
    # Load model
    processor, magma = get_magma_model(cfg.model_name)
    
    # Add the path to find openpi
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
    from openpi.third_party.mimicgen import mimicgen # Ensure mimicgen is imported

    # Create a mimicgen suite environment
    env = suite.make(
        env_name="RedBlueBlocks",
        has_renderer=False,  # Set to False initially to avoid display issues
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_heights=448,
        camera_widths=448,
        robots="Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
    )

    # Reset the environment
    obs = env.reset()


    # Get prompt for current task
    task = "point to the red block"
    prompt = get_magma_prompt(task, processor, magma.config)

    # Initialize task-specific counters
    task_episodes, task_successes = 0, 0
    
    step = 0
    rollout_images = []
    while step < 50 + cfg.num_steps_wait:
        if step < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action())
            step += 1
            continue
        image = obs['agentview_image']
        # flip the image vertically
        image = np.flip(image, axis=0)
        img = get_libero_obs(obs, resize_size=cfg.resolution)
        rollout_images.append(image)
        
        action = get_magma_action(magma, processor, img, prompt, cfg.task_suite_name)
        obs, reward, done, info = env.step(action.tolist())
        step += 1
        
        if done:
            task_successes += 1
            break
    
    task_episodes += 1
    save_rollout_video(rollout_images, os.path.join(cfg.save_dir, f"rollout_task_{cfg.task_suite_name}_seed_{cfg.seed}.mp4"), task)
    
    # Update total counters
    total_episodes += task_episodes
    total_successes += task_successes
    
    # Log final suite success rate
    suite_success_rate = float(total_successes) / float(total_episodes)
    print(f"Task suite success rate: {suite_success_rate}")
    
    return total_episodes, total_successes

if __name__ == "__main__":
    eval_mimicgen()