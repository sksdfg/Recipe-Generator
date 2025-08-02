import modal


cuda_version = "12.8.0"  
flavor = "devel"  
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

volume = modal.Volume.from_name("genai-results", create_if_missing=True)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "diffusers",
        "transformers",
        "torch",
        "torchvision",
        "datasets",
        "wandb",
        "bitsandbytes",
        "peft",
        "sentencepiece",
        "git+https://github.com/huggingface/diffusers.git",
        "ftfy",
        "opencv-python",
        "imageio",
        "imageio-ffmpeg"
    )
)

app = modal.App("video_generation",image=image)


@app.function(gpu="A100-80GB",timeout=86400, volumes={"/results": volume})
def run():
    import torch
    from diffusers.utils import export_to_video
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    flow_shift = 5.0 
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler
    pipe.to("cuda")

    prompt = """Simple Chicken and Rice Bake
1. Preheat the oven to 375°F (190°C).
2. In a large bowl, mix together the rice, salt, pepper, and paprika.
3. In a large skillet, heat the olive oil over medium-high heat. Add the chicken breasts and cook until browned on both sides, about 5-6 minutes per side.
4. In a 9x13 inch baking dish, arrange half of the rice mixture in the bottom of the dish.
5. Place the browned chicken breasts on top of the rice.
6. Add the remaining rice mixture around the chicken, making sure the chicken is mostly covered.
7. Cover the dish with aluminum foil and bake for 45 minutes.
8. Remove the foil and continue baking for an additional 15-20 minutes, or until the chicken is cooked through and the rice is tender.

give me the video for this dish"""

    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=720,
        width=1280,
        num_frames=81,
        guidance_scale=5.0,
        ).frames[0]
    export_to_video(output, "/results/output.mp4", fps=16)


if __name__ == "__main__":
    with app.run():
        run.call()
    


    