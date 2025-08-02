from gradio_client import Client
import os
import shutil
from datetime import datetime
import uuid

# Create images directory if it doesn't exist
images_dir = "images"
os.makedirs(images_dir, exist_ok=True)

client = Client("stabilityai/stable-diffusion-3-medium")
result = client.predict(
		prompt=" In a double boiler or a heatproof bowl set over a pot of simmering water, melt the chocolate, stirring occasionally.",
		negative_prompt="",
		seed=0,
		randomize_seed=True,
		width=1024,
		height=1024,
		guidance_scale=5,
		num_inference_steps=28,
		api_name="/infer"
)
l=[]
for i in result:
    l.append(i)
temp_image_path = l[0]
print(f"Temporary image path: {temp_image_path}")

# Extract the file extension
_, file_extension = os.path.splitext(temp_image_path)

# Create a unique filename with timestamp and UUID
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
filename = f"sd3_image_{timestamp}_{unique_id}{file_extension}"
saved_image_path = os.path.join(images_dir, filename)

# Copy the image from temp location to our images directory
shutil.copy2(temp_image_path, saved_image_path)

print(f"Image saved to: {saved_image_path}")


    