from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "Qwen/Qwen3-32B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

input("Model loaded. Press Enter to continue...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "What is 235 + 432?"

messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Starting generation")
time = datetime.now()

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=100,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

time = (datetime.now() - time).total_seconds()
print(f"Total time taken: {time} seconds")

print(f"{output_text}")
