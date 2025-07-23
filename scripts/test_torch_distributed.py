import os
from datetime import datetime
# WORLD_SIZE = 8
# RANK = os.environ['LOCAL_RANK']
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Qwen3ForCausalLM, AutoConfig
# import torch.distributed as dist
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
# os.environ['LOCAL_RANK'] = str(RANK)
# dist.init_process_group('nccl', rank=RANK, world_size=WORLD_SIZE)
rank = os.environ["LOCAL_RANK"]
model_name = "Qwen/Qwen3-32B"
tp_plan = {
    "model.layers.*.self_attn.q_proj": "colwise",
    "model.layers.*.self_attn.k_proj": "colwise",
    "model.layers.*.self_attn.v_proj": "colwise",
    "model.layers.*.self_attn.o_proj": "rowwise",
    "model.layers.*.mlp.up_proj": "colwise",
    "model.layers.*.mlp.gate_proj": "colwise",
    "model.layers.*.mlp.down_proj": "rowwise",
    "lm_head": "colwise_rep",
}
model = AutoModelForCausalLM.from_pretrained(model_name, tp_plan=tp_plan)
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "Can I help"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=f"cuda:{rank}")

if rank == "0":
    print("Starting generation")
    time = datetime.now()

output = pipe(prompt, max_length=10, max_new_tokens=10)

if rank == "0":
    time = (datetime.now() - time).total_seconds()
    print(output)
    print(f"Total time taken: {time} seconds")

# import os
# import torch
# import transformers
# import deepspeed
# local_rank = int(os.getenv("LOCAL_RANK", "0"))
# world_size = int(os.getenv("WORLD_SIZE", "1"))
# # create the model pipeline
# pipe = transformers.pipeline(task="text2text-generation", model="Qwen/Qwen3-32B-AWQ", device=local_rank)
# # Initialize the DeepSpeed-Inference engine
# pipe.model = deepspeed.init_inference(
#     pipe.model,
#     mp_size=world_size,
#     dtype=torch.float
# )
# output = pipe('Input String')