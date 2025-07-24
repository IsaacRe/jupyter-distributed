import torch
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
rank = os.environ["LOCAL_RANK"]
model_name = "Qwen/Qwen2-1.5B-Instruct" #"Qwen/Qwen3-32B"
# tp_plan = {
#     "model.layers.*.self_attn.q_proj": "colwise",
#     "model.layers.*.self_attn.k_proj": "colwise",
#     "model.layers.*.self_attn.v_proj": "colwise",
#     "model.layers.*.self_attn.o_proj": "rowwise",
#     "model.layers.*.mlp.up_proj": "colwise",
#     "model.layers.*.mlp.gate_proj": "colwise",
#     "model.layers.*.mlp.down_proj": "rowwise",
#     "lm_head": "colwise_rep",
# }
tp_plan = "auto"
model = AutoModelForCausalLM.from_pretrained(model_name, tp_plan=tp_plan)

device = f'cuda:{rank}'
max_allocated = torch.cuda.max_memory_allocated(device)
print(f"RANK {rank}: Maximum memory allocated on device {device}: {max_allocated / 1024**2:.2f} MB")

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

if rank == "0":
    print("\nStarting generation")
    time = datetime.now()

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")


if rank == "0":
    time = (datetime.now() - time).total_seconds()
    print(f"Total time taken: {time} seconds\n")

print(f"RANK {rank}: {output_text}")
