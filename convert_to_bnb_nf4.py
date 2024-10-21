import os
import shutil
import json
from transformers import AutoModelForVision2Seq, AutoTokenizer, BitsAndBytesConfig
import torch
import sys

# Define the quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Get model name from command line arguments
model_name = sys.argv[1]

# Load the model and tokenizer with the quantization configuration
model = AutoModelForVision2Seq.from_pretrained(model_name, quantization_config=quantization_config, low_cpu_mem_usage = True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Print the original model size
original_size = sum(p.numel() for p in model.parameters())
print(f"Original model size: {original_size / 1e6:.2f} million parameters")

# Create new model name with suffix
new_model_name = model_name + '-nf4'

# Save the quantized model with new name
model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)

# Modify config.json to remove the specified lines
with open(os.path.join(new_model_name, 'config.json'), 'r') as file:
    config_dict = json.load(file)

# Remove the specified fields
config_dict["quantization_config"].pop("_load_in_4bit", None)
config_dict["quantization_config"].pop("_load_in_8bit", None)
config_dict["quantization_config"].pop("quant_method", None)

# Write the modified config back to the file
with open(os.path.join(new_model_name, 'config.json'), 'w') as file:
    json.dump(config_dict, file, indent=4)

# Copy preprocessor_config.json and chat_template.json from input model to new model
shutil.copyfile(os.path.join(model_name, 'preprocessor_config.json'), os.path.join(new_model_name, 'preprocessor_config.json'))
shutil.copyfile(os.path.join(model_name, 'chat_template.json'), os.path.join(new_model_name, 'chat_template.json'))

print(f"Model has been quantized and saved as {new_model_name}. Config.json has been modified.")