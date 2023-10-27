import torch
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments as TransformerTrainingArguments,
)


class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        training_args: TransformerTrainingArguments,
        trainer_state: TrainerState,
        trainer_control: TrainerControl,
        **kwargs,
    ):
        if (trainer_state.global_step + 1) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(
                self.trainer.deepspeed
            )
            if self.trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(
                    training_args.output_dir, state_dict=state_dict
                )
            self.trainer.accelerator.wait_for_everyone()
        return trainer_control


def create_and_prepare_model(
    model_id: str, training_args: TransformerTrainingArguments, script_args
):
    # Quantization
    bits_and_bytes_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    pretrained_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bits_and_bytes_config,
        use_cache=not training_args.gradient_checkpointing,
        use_flash_attention_2=script_args.use_flash_attn,
    )
    print("------- Model loaded -------")

    target_linear_modules = find_all_linear_names(pretrained_model)

    # Lora config
    lora_configuration = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_linear_modules,
    )

    if training_args.gradient_checkpointing:
        pretrained_model.gradient_checkpointing_enable()

    print("------- Pre-processing model for PEFT -------")
    for name, module in pretrained_model.named_modules():
        if (
            isinstance(module, LoraLayer)
            or "norm" in name
            or any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"])
        ):
            module = module.to(torch.bfloat16)

    print("------- Initializing PEFT model -------")
    peft_model = get_peft_model(pretrained_model, lora_configuration)

    peft_model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    return peft_model, lora_configuration, tokenizer


def find_all_linear_names(model):
    linear_class = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_class):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)
