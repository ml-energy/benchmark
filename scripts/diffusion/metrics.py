from functools import partial

from datasets import load_dataset
import torch
from torchmetrics.functional.multimodal import clip_score


def load_prompts(num_prompts, batch_size):
    """Generate prompts for CLIP Score metric.

    Args:
        num_prompts (int): number of prompts to generate.
            If num_prompts == 0, returns all prompts instead.
        batch_size (int): batch size for prompts

    Returns:
        A tuple (prompts, batched_prompts) where prompts is a list of prompts
        of length num_prompts (if num_prompts != 0) or the list of all prompts
        (if num_prompts == 0), and batched_prompts is the list of prompts,
        batched into chunks of size batch_size each.
    """
    prompts = load_dataset("nateraw/parti-prompts", split="train")
    if num_prompts == 0:
        num_prompts = len(prompts)
    else:
        prompts = prompts.shuffle()
    prompts = prompts[:num_prompts]["Prompt"]
    batched_prompts = [
        prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)
    ]
    if len(batched_prompts[-1]) < batch_size:
        batched_prompts = batched_prompts[:-1]
    prompts = [prompt for batch in batched_prompts for prompt in batch]
    return prompts, batched_prompts


def calculate_clip_score(images, prompts):
    """Calculate CLIP Score metric.

    Args:
        images (np.ndarray): array of images
        prompts (list): list of prompts, assumes same size as images

    Returns:
        The clip score across all images and prompts as a float.
    """
    clip_score_fn = partial(
        clip_score, model_name_or_path="openai/clip-vit-base-patch16"
    )
    images_int = (images * 255).astype("uint8")
    clip = clip_score_fn(
        torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts
    ).detach()
    return float(clip)
