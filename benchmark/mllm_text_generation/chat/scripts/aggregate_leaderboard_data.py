import json
from glob import glob
from pathlib import Path

import tyro


SATURATION_THRESHOLD = 0.95

def main(result_dir: Path, output_dir: Path) -> None:
    print(f"{result_dir} -> {output_dir}")

    for model_dir in sorted(glob(f"{result_dir}/*/*")):
        model_name = "/".join(model_dir.split("/")[-2:])
        print(f"  {model_name}")
        (output_dir / model_name).mkdir(parents=True, exist_ok=True)

        # Gather all results files.
        results = sorted(glob(f"{model_dir}/vllm+*+results.json"))

        # Gather all stats files. Skip if stats file is missing.
        files: list[tuple[str, str]] = []
        for result_file in results:
            stats_file = result_file.replace("+results.json", "+stats.json")
            if Path(stats_file).exists():
                files.append((result_file, stats_file))

        # Produce one JSON file per (results, stats) pair.
        for result_path, stats_path in files:
            with open(result_path) as f:
                result_data = json.load(f)
            with open(stats_path) as f:
                stats_data = json.load(f)

            # Final output data.
            data = {}

            # Derive metrics.
            pp = len(stats_data["steady_state"])
            ss_total_time = max(node["time"] for node in stats_data["steady_state"])
            ss_total_energy = sum(sum(node["energy"].values()) for node in stats_data["steady_state"])
            ss_end_iter = next(filter(lambda iq: iq[1] == 0, enumerate(stats_data["num_waiting_sys"][2:])))[0] + 2
            ss_total_output_tokens = sum(stats_data["num_generation_tokens_iter"][:ss_end_iter])
            average_output_length = result_data["total_completion_tokens"] / result_data["num_requests"]
            tpot = []
            for iter_tpot in stats_data["time_per_output_tokens_iter"]:
                if iter_tpot:
                    tpot.append(sum(iter_tpot) / len(iter_tpot))

            # Actual fields.
            data["Model"] = result_data["model"]
            data["GPU"] = result_data["gpu_model"]
            data["TP"] = result_data["num_gpus"]
            data["PP"] = pp
            data["Energy/req (J)"] = ss_total_energy / ss_total_output_tokens * average_output_length
            data["Avg TPOT (s)"] = sum(tpot) / len(tpot)
            data["Token tput (tok/s)"] = ss_total_output_tokens / ss_total_time
            data["Avg Output Tokens"] = average_output_length
            data["Avg BS (reqs)"] = sum(stats_data["num_running_sys"][:ss_end_iter]) / ss_end_iter
            data["Max BS (reqs)"] = result_data["max_num_seqs"] * pp

            # Skip saturated runs.
            if data["Max BS (reqs)"] * SATURATION_THRESHOLD >= data["Avg BS (reqs)"]:
                continue

            # Dump output data.
            filename = f"bs{result_data['max_num_seqs']}+tp{data['TP']}+pp{data['PP']}.json"
            output_path = output_dir / model_name / filename
            json.dump(data, open(output_path, "w"), indent=2)


if __name__ == "__main__":
    tyro.cli(main)
