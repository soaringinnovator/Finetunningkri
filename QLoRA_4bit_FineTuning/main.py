from install_requirements import *
from import_libraries import *
from load_dataset import load_dataset_func
from configure_model import configure_model
from fine_tune_model import fine_tune_model
from save_model import save_trained_model
from run_tensorboard import run_tensorboard
from text_generation import generate_text

# Step 1: Set up parameters
model_name = "your/model_name_here"
dataset_name = "your/dataset_name_here"
new_model_name = "fine-tuned-model-name"
use_4bit = True
bnb_4bit_quant_type = "nf4"
device_map = {"": 0}
output_dir = "./results"

# Step 2: Load dataset
dataset = load_dataset_func(dataset_name)

# Step 3: Configure model
model = configure_model(model_name, use_4bit, bnb_4bit_quant_type, device_map)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 4: Fine-tune model
training_arguments = TrainingArguments(output_dir=output_dir, ...)
peft_config = LoraConfig(...)  # Define your PEFT config here
trainer = fine_tune_model(model, dataset, tokenizer, training_arguments, peft_config)

# Step 5: Save the model
save_trained_model(trainer, new_model_name)

# Step 6: Run TensorBoard
run_tensorboard(output_dir)

# Step 7: Text generation example
prompt = "What is a large language model?"
generated_text = generate_text(model, tokenizer, f"[INST] {prompt} [/INST]")
print(generated_text)
