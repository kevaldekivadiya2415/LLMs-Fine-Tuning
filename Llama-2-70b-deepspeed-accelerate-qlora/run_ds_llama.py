"""
This script is used for training a model using Hugging Face's transformers library. It supports the use of QLoRA and Flash attention for training.

"""

from dataclasses import dataclass, field
from typing import Optional, cast

import torch
from transformers import HfArgumentParser, TrainingArguments
from utils.peft_utils import SaveDeepSpeedPeftModelCallback, create_and_prepare_model
from datasets import load_dataset
from trl import SFTTrainer


@dataclass
class ScriptArguments:
    """
    Additional arguments for training, which are not part of TrainingArguments.
    """

    model_id: str = field(
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    lora_alpha: Optional[int] = field(default=64)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=16)
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )


def format_dolly(sample):
    # Extract the 'instruction', 'context', and 'response' from the 'sample' dictionary
    instruction = f"### Instruction\n{sample['instruction']}"
    # Check if 'context' is not an empty string
    context = f"### Context\n{sample['context']}" if sample["context"] else None

    # Extract the 'response' from the 'sample' dictionary
    response = f"### Answer\n{sample['response']}"

    # Join the parts together with a double line break, omitting 'None' values
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])

    # Return the formatted prompt
    return prompt


def training_function(script_args: ScriptArguments, training_args: TrainingArguments):
    """
    Function to train the custom model using the provided arguments.

    Args:
    script_args (ScriptArguments): Additional script arguments.
    training_args (TrainingArguments): Training arguments.

    """

    # Load dataset
    dataset = load_dataset(script_args.dataset_name, split="train")

    # Load and create PEFT model
    model, peft_config, tokenizer = create_and_prepare_model(
        script_args.model_id, training_args, script_args
    )
    model.config.use_cache = False

    # Create trainer and add callbacks
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        formatting_func=format_dolly,
        packing=True,
        max_seq_length=4096,
    )

    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()
    trainer.add_callback(
        SaveDeepSpeedPeftModelCallback(trainer, save_steps=training_args.save_steps)
    )

    # Start training
    trainer.train()

    # Save model on main process
    trainer.accelerator.wait_for_everyone()
    state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
    if trainer.accelerator.is_main_process:
        unwrapped_model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    trainer.accelerator.wait_for_everyone()

    # Save everything else on the main process
    if trainer.args.process_index == 0:
        trainer.model.save_pretrained(training_args.output_dir, safe_serialization=True)

        # save tokenizer
        tokenizer.save_pretrained(training_args.output_dir)


def main():
    parser = HfArgumentParser([ScriptArguments, TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()
    script_args = cast(ScriptArguments, script_args)
    training_args = cast(TrainingArguments, training_args)

    training_function(script_args, training_args)


if __name__ == "__main__":
    main()
