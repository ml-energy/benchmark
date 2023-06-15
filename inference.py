"""Perform inference of one model on one input prompt and measure time and energy."""

from __future__ import annotations

from typing import Literal

import tyro
import rich
import torch
from fastchat.serve.inference import generate_stream
from fastchat.model.model_adapter import load_model, get_conversation_template
from zeus.monitor import ZeusMonitor

SYSTEM_PROMPTS = {
    "chat": (
        "A chat between a human user (prompter) and an artificial intelligence (AI) assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    "chat-concise": (
        "A chat between a human user (prompter) and an artificial intelligence (AI) assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "The assistnat's answers are concise but high-quality."
    ),
    "instruct": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    ),
    "instruct-concise": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        "The response should be concise but high-quality."
    ),
}


def main(
    model_path: str,
    input_prompt: str,
    device_index: int = 0,
    task: Literal[tuple(SYSTEM_PROMPTS)] = "chat",  # type: ignore
    load_8bit: bool = False,
    temperature: float = 0.7,
    repitition_penalty: float = 1.0,
    max_new_tokens: int = 512,
) -> None:
    """Run the main routine.

    Code structure is based on
    https://github.com/lm-sys/FastChat/blob/57dea54055/fastchat/serve/inference.py#L249

    Args:
        model_path: Path to or Huggingface Hub Id of the model.
        input_prompt: Input prompt to use for inference.
        device_index: Index of the GPU to use for inference.
        task: Type of task to perform inference on.
        load_8bit: Whether to load the model in 8-bit mode.
        temperature: Temperature to use for sampling.
        repitition_penalty: Repitition penalty to use for the model.
        max_new_tokens: Maximum numbers of tokens to generate, ignoring the prompt.
    """
    # NOTE(JW): ChatGLM is implemented as a special case in FastChat inference.
    # Also, it's primarily a model that's fine-tuned for Chinese, so it doesn't
    # make sense to prompt it in English and talk about its verbosity.
    if "chatglm" in model_path.lower():
        raise ValueError("ChatGLM is not supported.")

    # Set the device.
    torch.cuda.set_device(f"cuda:{device_index}")

    # Load the model (Huggingface PyTorch) and tokenizer (Huggingface).
    model, tokenizer = load_model(
        model_path=model_path,
        device="cuda",
        num_gpus=1,
        max_gpu_memory=None,
        load_8bit=load_8bit,
        cpu_offloading=False,
        gptq_config=None,
        debug=False,
    )

    # Chats are accumulated in a conversation helper object.
    conv = get_conversation_template(model_path)

    # Standardize the system prompt for every model.
    conv.system = SYSTEM_PROMPTS[task]
    conv.messages = []
    conv.offset = 0

    # Construct the input prompt.
    conv.append_message(conv.roles[0], input_prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    # Generate the ouptut from the model.
    gen_params = {
        "model": model_path,
        "prompt": prompt,
        "temperature": temperature,
        "repitition_penalty": repitition_penalty,
        "max_new_tokens": max_new_tokens,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }
    output_stream = generate_stream(model, tokenizer, gen_params, device="cuda")
    output = {}

    # Inference and measurement!
    monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
    monitor.begin_window("inference")
    for output in output_stream:
        pass
    measurements = monitor.end_window("inference")
    
    # Print the input and output.
    rich.print(f"\n[u]Prompt[/u]:\n{prompt.strip()}\n")
    output_text = output["text"]
    rich.print(f"\n[u]Response[/u]:\n{output_text.strip()}\n")

    # Print numbers.
    num_tokens = len(tokenizer.encode(output_text))
    rich.print(measurements)
    rich.print(f"Number of tokens: {num_tokens}")
    rich.print(f"Tokens per seconds: {num_tokens / measurements.time:.2f}")
    rich.print(f"Joules per token: {measurements.total_energy / num_tokens:.2f}")
    rich.print(f"Average power consumption: {measurements.total_energy / measurements.time:.2f}")


if __name__ == "__main__":
    tyro.cli(main)
