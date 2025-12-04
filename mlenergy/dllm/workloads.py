from pathlib import Path

from mlenergy.llm.workloads import (
    LMArenaChat,
)


class LMArenaChatDLLM(LMArenaChat):
    """Workload using the LMArena human preference dataset.

    Simplified version of LMArenaChat for benchmarking fast-dllm. There are some
    assumptions of WorkloadConfig that is coupled with vLLM such as max_num_seqs.
    As they are required as arguments, we don't use them
    """

    batch_size: int = 1
    block_size: int = 128
    num_steps: int = 32

    def to_filename_parts(self) -> list[str]:
        return [
            "lmarena_chat",
            str(self.num_requests) + "req",
            str(self.seed) + "seed",
            str(self.batch_size) + "batch_size",
        ]


def default_lmarena_chat_dllm() -> LMArenaChatDLLM:
    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    max_num_seqs: int = 2
    num_requests: int = 4
    model_id: str = model_id
    gpu_model: str = "H100"
    base_dir: Path = Path("run/llm/lmarena") / model_id

    return LMArenaChatDLLM(
        base_dir=base_dir,
        num_requests=num_requests,
        model_id=model_id,
        max_num_seqs=max_num_seqs,
        gpu_model=gpu_model,
    )


if __name__ == "__main__":
    """Testing"""

    model_ids = ["GSAI-ML/LLaDA-8B-Instruct"]

    work = LMArenaChatDLLM(
        base_dir=Path("run/llm/lmarena") / model_ids[0],
        num_requests=32,
        model_id=model_ids[0],
        max_num_seqs=16,  # Placeholder, not used in this workload
        batch_size=2,
    )

    input_requests = work.load_requests()
    print(input_requests[0].prompt)
