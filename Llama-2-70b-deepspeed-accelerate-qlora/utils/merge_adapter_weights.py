from dataclasses import dataclass, field
from typing import Optional
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, HfArgumentParser


@dataclass
class MergeScriptArguments:
    peft_model_id: str = field(metadata={"help": "Model ID or path to the model"})
    output_dir: Optional[str] = field(
        default="merged-weights",
        metadata={"help": "Directory to save the merged model"},
    )
    save_tokenizer: Optional[bool] = field(
        default=True, metadata={"help": "Specify whether to save the tokenizer"}
    )


parser = HfArgumentParser(MergeScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

merged_model = AutoPeftModelForCausalLM.from_pretrained(
    script_args.peft_model_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained(
    script_args.output_dir, safe_serialization=True, max_shard_size="10GB"
)

if script_args.save_tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(script_args.peft_model_id)
    tokenizer.save_pretrained(script_args.output_dir)
