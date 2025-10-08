import torch
import pyrealsense2 as rs
import numpy as np
import cv2# resize image down to 224x224 and ensure it's RGB
from PIL import Image
import sys
import os
from magma.processing_magma import MagmaProcessor
from magma.modeling_magma import MagmaForCausalLM
from transformers import AutoModelForCausalLM, AutoProcessor

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_video(frames, filename="output.mp4", fps=30):
    """
    Save a list of frames (numpy arrays) as a video file
    """
    if len(frames) == 0:
        print("No frames to save.")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {filename}")

def capture_single_frame(discard_initial_frames=50):
    """
    Capture a single frame from the RealSense camera and return it as a PIL Image
    """
    try:
        # Initialize RealSense camera
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        pipe.start(cfg)
        print("RealSense camera initialized successfully")
        
        # discard initial frames to allow auto-exposure to adjust
        for _ in range(discard_initial_frames):
            pipe.wait_for_frames()

        # Capture a frame
        print("Capturing frame from camera...")
        frames = pipe.wait_for_frames()
        color = frames.get_color_frame()
        
        if not color:
            print("Failed to capture frame")
            pipe.stop()
            return None
            
        # Convert to numpy array and then to PIL Image
        rgb = np.asanyarray(color.get_data())  # This is BGR from RealSense
        

        # Stop the camera
        pipe.stop()
        print("Frame captured successfully")
        
        # Return both BGR (for cv2 saving) and RGB (for model)
        return rgb
        
    except Exception as e:
        print(f"Failed to capture frame from camera: {e}")
        return None

if __name__ == "__main__":
    # Load Magma model
    print("Loading Magma model...")
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Magma-8B",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
    processor = MagmaProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)
    model.to("cuda")
    print("Model loaded successfully")

    # Capture frame from camera
    rgb_image = capture_single_frame()


    if rgb_image is None:
        print("Could not capture frame from camera. Exiting.")
        sys.exit(1)


    # Save the captured frame for verification using correct color format
    bgr_path = os.path.join(SCRIPT_DIR, "captured_frame_bgr.png")
    rgb_path = os.path.join(SCRIPT_DIR, "captured_frame_rgb.png")
    pil_path = os.path.join(SCRIPT_DIR, "captured_frame_pil.png")

    # Save RGB version (convert RGB to BGR for cv2.imwrite, but keep RGB data)
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))  # Convert RGB back to BGR for saving
    print(f"Captured frames saved in {SCRIPT_DIR}:")
    print(f"  - BGR version: {bgr_path}")
    print(f"  - RGB version: {rgb_path}")
    print(f"RGB image size: {rgb_image.shape}")

    # Let's also check the color values in a small region to verify the conversion
    print("Color values in top-left corner (now properly RGB):")
    print(f"RGB [R,G,B]: {rgb_image[10:15, 10:15, :].mean(axis=(0,1))}")

    # Also save a PIL version to double-check
    pil_image = Image.fromarray(rgb_image)
    # make it RGB explicitly
    pil_image = pil_image.convert("RGB")
    pil_image.save(pil_path)
    print(f"  - PIL version: {pil_path}")

    # Process the image with Magma
    convs = [
        {"role": "system", "content": "You are an agent that can see, talk and act."},
        {"role": "user", "content": "<image_start><image><image_end>\nDescribe the objects and their colors in the scene."},
    ]
    prompt = processor.tokenizer.apply_chat_template(
            convs,
            tokenize=False,
            add_generation_prompt=True
        )
    # Use the correct processor format - pass text and images separately
    print(f"Input image shape before processor: {rgb_image.shape}")
    print(f"Input image dtype: {rgb_image.dtype}")
    print(f"Input image value range: [{rgb_image.min()}, {rgb_image.max()}]")

    inputs = processor(images=[rgb_image], texts=prompt, return_tensors="pt")

    # Check what the processor did to the image
    print(f"Processed pixel_values shape: {inputs['pixel_values'].shape}")
    print(f"Processed pixel_values dtype: {inputs['pixel_values'].dtype}")
    print(f"Processed pixel_values range: [{inputs['pixel_values'].min():.3f}, {inputs['pixel_values'].max():.3f}]")


    # Use the PIL-processed inputs for consistency
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)   
    inputs = inputs.to("cuda").to(torch.bfloat16)

    # Move inputs to CUDA
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].to("cuda")

    print("Running Magma inference on captured frame...")
    print(f"Input keys: {list(inputs.keys())}")
    for key, value in inputs.items():
        if torch.is_tensor(value):
            print(f"{key} shape: {value.shape}")

    generation_args = {
            "max_new_tokens": 500,  # Reduced for stability
            "temperature": 0.0,  # Deterministic generation
            "do_sample": False,  # Use greedy decoding
            "pad_token_id": processor.tokenizer.eos_token_id,  # Set pad token
        }

    try:
        with torch.inference_mode():
            generate_ids = model.generate(**inputs, **generation_args)

        generate_ids = generate_ids[:, inputs["input_ids"].shape[-1] :]
        response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()

        print("=" * 50)
        print("MAGMA RESPONSE:")
        print("=" * 50)
        print(response)
        print("=" * 50)

    except Exception as e:
        print(f"Error during model generation: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

