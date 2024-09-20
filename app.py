"""Gradio app for the ML.ENERGY leaderboard.

Everything is in a single file. Search for `gr.Blocks` to find the place
where UI elements are actually defined.
"""

from __future__ import annotations

import copy
import json
import random
import yaml
import requests
import itertools
import contextlib
import argparse
import os
from pathlib import Path
from abc import abstractmethod
from typing import Literal, Any
from dateutil import parser, tz

import numpy as np
import gradio as gr
import pandas as pd

from spitfight.colosseum.client import ControllerClient

COLOSSEUM_UP = True
COLOSSEUM_DOWN_MESSAGE = f"<br/><h2 style='text-align: center'>The Colosseum is currently down for maintenance.</h2>"
COLOSSUMM_YOUTUBE_DEMO_EMBED_HTML = '<div style="width: 100%; min-width: 400px;"><div style="position: relative; width: 100%; overflow: hidden; padding-top: 56.25%"><p><iframe width="560" height="315" style="margin: auto; position: absolute; top: 0; left: 0; right: 0; width: 100%; height: 100%; border: none;" src="https://www.youtube.com/embed/tvNM_gLffFs?si=rW1-10pt5BffJEGH" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe><p></div></div>'


class TableManager:
    """Manages the data for the leaderboard tables for tasks."""

    def __init__(self, data_dir: str) -> None:
        """Load leaderboard data from files in `data_dir`.

        Expected directory structure: `data_dir/gpu_model`.
        Inside the innermost (GPU) directory, there should be:
        - `models.json`: JSON file that maps huggingface model IDs to model info.
              Some models listed in this file may not have benchmark results.
        - `model_org/model_name/*.json`: JSON files containing the benchmark results.
        """
        self.data_dir = Path(data_dir)

    def __str__(self) -> str:
        return f"{self.__class__}(data_dir={self.data_dir})"

    def _wrap_model_name(self, url: str, model_name: str) -> str:
        """Wrap the model name in an HTML anchor."""
        return f'<a style="text-decoration: underline; text-decoration-style: dotted" target="_blank" href="{url}">{model_name}</a>'

    def _unwrap_model_name(self, model_name: str) -> str:
        """Unwrap the model name from an HTML anchor."""
        return model_name.split(">")[1].split("<")[0]

    @abstractmethod
    def get_tab_name(self) -> str:
        """Return the name of the leaderboard."""

    @abstractmethod
    def get_intro_text(self) -> str:
        """Return the introduction text to be inserted above the table."""

    @abstractmethod
    def get_detail_text(self, detail_mode: bool) -> str:
        """Return the detail text chunk to be inserted below the table."""

    def get_benchmark_checkboxes(self) -> dict[str, list[str]]:
        """Return data for the benchmark selection checkboxes."""
        return {}

    def get_benchmark_sliders(self) -> dict[str, tuple[float, float, float, float]]:
        """Return data for the benchmark selection sliders.

        Dictionary values are tuples of the form (min, max, step, default).
        """
        return {}

    @abstractmethod
    def get_all_models(self) -> list[str]:
        """Return all available models."""

    @abstractmethod
    def set_filter_get_df(self, detail_mode: bool, *filters) -> pd.DataFrame:
        """Set the current set of filters and return the filtered DataFrame."""


class LLMTableManager(TableManager):
    def __init__(self, data_dir: str, task_name: str) -> None:
        """Load leaderboard data from files in `data_dir`.

        Under `data_dir`, there should be:
        - `models.json`: JSON file that maps huggingface model IDs to model info.
              Some models listed in this file may not have benchmark results.
        - `schema.yaml`: YAML file containing the schema of the benchmark.

        Then, benchmark data files are nested under `data_dir` according to the schema.
        One directory hierarchy for each choice in the schema and then two more -- the
        model's HuggingFace hub organization and the model name.
        """
        super().__init__(data_dir)

        self.task_name = task_name

        # Read in the data into a Pandas DataFrame.
        # Important: The ordering `self.schema` determines the directory structure.
        self.schema = yaml.safe_load(open(self.data_dir / "schema.yaml"))
        models: dict[str, dict[str, Any]] = json.load(
            open(self.data_dir / "models.json")
        )
        res_df = pd.DataFrame()
        for choice in itertools.product(*self.schema.values()):
            result_dir = self.data_dir / "/".join(choice)
            with contextlib.suppress(FileNotFoundError):
                for model_id, model_info in models.items():
                    for file in (result_dir / model_id).glob("*.json"):
                        model_df = pd.DataFrame([json.load(open(file))])
                        # Sanity checks and standardization of schema values.
                        assert model_df["Model"].iloc[0] == model_id
                        for key, val in zip(self.schema.keys(), choice):
                            assert (
                                str(val).lower() in str(model_df[key].iloc[0]).lower()
                            )
                            model_df[key] = val
                        # Format the model name as an HTML anchor.
                        model_df["Model"] = self._wrap_model_name(model_info["url"], model_info["nickname"])
                        model_df["Params (B)"] = model_info["params"]
                        res_df = pd.concat([res_df, model_df])

        if res_df.empty:
            raise ValueError(
                f"No benchmark JSON files were read from {self.data_dir=}."
            )

        # Order columns
        columns = res_df.columns.to_list()
        cols_to_order = ["Model", "Params (B)"]
        cols_to_order.extend(self.schema.keys())
        columns = cols_to_order + [col for col in columns if col not in cols_to_order]
        res_df = res_df[columns]

        # Order rows
        res_df = res_df.sort_values(by=["Model", *self.schema.keys(), "Energy/req (J)"])

        self.full_df = res_df.round(2)

        # We need to set the default view separately when `gr.State` is forked.
        self.set_filter_get_df(detail_mode=False)

    def get_benchmark_checkboxes(self) -> dict[str, list[str]]:
        return self.schema

    def get_benchmark_sliders(self) -> dict[str, tuple[float, float, float, float]]:
        return {"Target Average TPOT (Time Per Output Token) (s)": (0.0, 0.5, 0.01, 0.2)}

    def get_all_models(self) -> list[str]:
        return self.full_df["Model"].apply(self._unwrap_model_name).unique().tolist()

    def set_filter_get_df(self, detail_mode: bool, *filters) -> pd.DataFrame:
        """Set the current set of filters and return the filtered DataFrame.

        Filters can either be completely empty, or be a concatenated list of
        choices from all checkboxes and all sliders.
        """
        # If the filter is empty, we default to the first choice for each checkbox.
        if not filters:
            checkboxes = [choices[:1] for choices in self.schema.values()]
            sliders = [slider[3] for slider in self.get_benchmark_sliders().values()]
            filters = checkboxes + sliders

        index = np.full(len(self.full_df), True)
        # Checkboxes
        for setup, choice in zip(self.schema, filters):
            index = index & self.full_df[setup].isin(choice)
        cur_df = self.full_df.loc[index]

        # Sliders (We just have TPOT for now.)
        # For each `Model`, we want to first filter out rows whose `Avg TPOT (s)` is greater than the slider value.
        # Finally, only just leave the row whose `Energy/req (J)` is the smallest.
        tpot_slo = filters[-1]
        cur_df = (
            cur_df
                .groupby("Model")[cur_df.columns]
                .apply(lambda x: x[x["Avg TPOT (s)"] <= tpot_slo], include_groups=True)
                .sort_values(by="Energy/req (J)")
                .reset_index(drop=True)
                .groupby("Model")
                .head(1)
        )

        if not detail_mode:
            core_columns = ["Model", "Params (B)", "GPU", "Energy/req (J)"]
            readable_name_mapping = {
                "Params (B)": "Parameters (Billions)",
                "GPU": "GPU model",
                "Energy/req (J)": "Energy per response (Joules)",
            }
            cur_df = cur_df[core_columns].rename(columns=readable_name_mapping)

        return cur_df


class LLMChatTableManager(LLMTableManager):
    """LLM table manager for chat tasks."""

    def get_tab_name(self) -> str:
        return "LLM Chat"

    def get_intro_text(self) -> str:
        text = """
            <h2>How much energy do GenAI models consume?</h2>

            <h3>LLM chatbot response generation</h3>

            <p style="font-size: 16px">
            Large language models (LLMs), especially the instruction-tuned ones, can generate human-like responses to chat prompts.
            Using <a href="https://ml.energy/zeus">Zeus</a> for energy measurement, we created a leaderboard for LLM chat energy consumption.
            </p>

            <p style="font-size: 16px">
            More models will be added over time. Stay tuned!
            </p>
            """
        return text

    def get_detail_text(self, detail_mode: bool) -> str:
        if detail_mode:
            text = """
                **TPOT (Time Per Output Token)** is the time between each token generated by LLMs as part of their response.
                An average TPOT of 0.20 seconds roughly corresponds to a person reading at 240 words per minute and assuming one word is 1.3 tokens on average.
                You can tweak the TPOT slider to adjust the target average TPOT for the models.

                Each row corresponds to one model, given a constraint on the maximum average TPOT.
                If more than one GPU types were chosen, the row shows results from the GPU with the lowest energy consumption per request.

                Columns
                - **Model**: The name of the model.
                - **Params (B)**: Number of parameters in the model.
                - **GPU**: Name of the GPU model used for benchmarking.
                - **TP**: Tensor parallelism degree.
                - **PP**: Pipeline parallelism degree. (TP * PP is the total number of GPUs used.)
                - **Energy/req (J)**: Energy consumed per request in Joules.
                - **Avg TPOT (s)**: Average time per output token in seconds.
                - **Token tput (toks/s)**: Average number of tokens generated by the engine per second.
                - **Avg Output Tokens**: Average number of output tokens in the LLM's response.
                - **Avg BS**: Average batch size of the serving engine over time.
                - **Max BS**: Maximum batch size configuration of the serving engine.

                For more detailed information, please take a look at the **About** tab.
                """
        else:
            text = """
                Columns
                - **Model**: The name of the model.
                - **Parameters (Billions)**: Number of parameters in the model. This is the size of the model.
                - **GPU model**: Name of the GPU model used for benchmarking.
                - **Energy per response (Joules)**: Energy consumed for each LLM response in Joules.

                Checking "Show more technical details" above the table will reveal more detailed columns.
                Also, for more detailed information, please take a look at the **About** tab.
                """

        return text



class LLMCodeTableManager(LLMTableManager):
    """LLM table manager for coding tasks."""

    def get_tab_name(self) -> str:
        return "LLM Code"

    def get_intro_text(self) -> str:
        text = """
            <h2>How much energy do GenAI models consume?</h2>

            <h3>LLM code generation</h3>

            <p style="font-size: 16px">
            Large language models (LLMs) are also capable of generating code.
            Using <a href="https://ml.energy/zeus">Zeus</a> for energy measurement, we created a leaderboard for the energy consumption of LLMs specifically trained for code generation.
            </p>

            <p style="font-size: 16px">
            More models will be added over time. Stay tuned!
            </p>
            """
        return text

    def get_detail_text(self, detail_mode: bool) -> str:
        if detail_mode:
            text = """
                **TPOT (Time Per Output Token)** is the time between each token generated by LLMs as part of their response.
                An average TPOT of 0.20 seconds roughly corresponds to a person reading at 240 words per minute and assuming one word is 1.3 tokens on average.
                You can tweak the TPOT slider to adjust the target average TPOT for the models.

                Each row corresponds to one model, given a constraint on the maximum average TPOT.
                If more than one GPU types were chosen, the row shows results from the GPU with the lowest energy consumption per request.

                Columns
                - **Model**: The name of the model.
                - **Params (B)**: Number of parameters in the model.
                - **GPU**: Name of the GPU model used for benchmarking.
                - **TP**: Tensor parallelism degree.
                - **PP**: Pipeline parallelism degree. (TP * PP is the total number of GPUs used.)
                - **Energy/req (J)**: Energy consumed per request in Joules.
                - **Avg TPOT (s)**: Average time per output token in seconds.
                - **Token tput (toks/s)**: Average number of tokens generated by the engine per second.
                - **Avg Output Tokens**: Average number of output tokens in the LLM's response.
                - **Avg BS**: Average batch size of the serving engine over time.
                - **Max BS**: Maximum batch size configuration of the serving engine.

                For more detailed information, please take a look at the **About** tab.
                """
        else:
            text = """
                Columns
                - **Model**: The name of the model.
                - **Parameters (Billions)**: Number of parameters in the model. This is the size of the model.
                - **GPU model**: Name of the GPU model used for benchmarking.
                - **Energy per response (Joules)**: Energy consumed for each LLM response in Joules.

                Checking "Show more technical details" above the table will reveal more detailed columns.
                Also, for more detailed information, please take a look at the **About** tab.
                """

        return text


class VLMChatTableManager(LLMTableManager):
    """VLM table manager for chat tasks."""

    def get_tab_name(self) -> str:
        return "VLM Visual Chat"

    def get_intro_text(self) -> str:
        text = """
            <h2>How much energy do GenAI models consume?</h2>

            <h3>VLM visual chatbot response generation</h3>

            <p style="font-size: 16px">
            Vision language models (VLMs) are large language models that can understand images along with text and generate human-like responses to chat prompts with images.
            Using <a href="https://ml.energy/zeus">Zeus</a> for energy measurement, we created a leaderboard for VLM chat energy consumption.
            </p>

            <p style="font-size: 16px">
            More models will be added over time. Stay tuned!
            </p>
            """
        return text

    def get_detail_text(self, detail_mode: bool) -> str:
        if detail_mode:
            text = """
                **TPOT (Time Per Output Token)** is the time between each token generated by LLMs as part of their response.
                An average TPOT of 0.20 seconds roughly corresponds to a person reading at 240 words per minute and assuming one word is 1.3 tokens on average.
                You can tweak the TPOT slider to adjust the target average TPOT for the models.

                Each row corresponds to one model, given a constraint on the maximum average TPOT.
                If more than one GPU types were chosen, the row shows results from the GPU with the lowest energy consumption per request.

                Columns
                - **Model**: The name of the model.
                - **Params (B)**: Number of parameters in the model.
                - **GPU**: Name of the GPU model used for benchmarking.
                - **TP**: Tensor parallelism degree.
                - **PP**: Pipeline parallelism degree. (TP * PP is the total number of GPUs used.)
                - **Energy/req (J)**: Energy consumed per request in Joules.
                - **Avg TPOT (s)**: Average time per output token in seconds.
                - **Token tput (toks/s)**: Average number of tokens generated by the engine per second.
                - **Avg Output Tokens**: Average number of output tokens in the LLM's response.
                - **Avg BS**: Average batch size of the serving engine over time.
                - **Max BS**: Maximum batch size configuration of the serving engine.

                For more detailed information, please take a look at the **About** tab.
                """
        else:
            text = """
                Columns
                - **Model**: The name of the model.
                - **Parameters (Billions)**: Number of parameters in the model. This is the size of the model.
                - **GPU model**: Name of the GPU model used for benchmarking.
                - **Energy per response (Joules)**: Energy consumed for each LLM response in Joules.

                Checking "Show more technical details" above the table will reveal more detailed columns.
                Also, for more detailed information, please take a look at the **About** tab.
                """

        return text


class DiffusionTableManager(TableManager):
    def __init__(self, data_dir: str, task_name: str) -> None:
        """Load leaderboard data from files in `data_dir`.

        Under `data_dir`, there should be:
        - `models.json`: JSON file that maps huggingface model IDs to model info.
              Some models listed in this file may not have benchmark results.
        - `schema.yaml`: YAML file containing the schema of the benchmark.

        Then, benchmark data files are nested under `data_dir` according to the schema.
        One directory hierarchy for each choice in the schema and then two more -- the
        model's HuggingFace hub organization and the model name.
        """
        super().__init__(data_dir)

        self.task_name = task_name

        if "to video" in task_name.lower():
            self.energy_col = "Energy/video (J)"
            self.energy_col_readable = "Energy per video (Joules)"
        elif "to image" in task_name.lower():
            self.energy_col = "Energy/image (J)"
            self.energy_col_readable = "Energy per image (Joules)"
        else:
            raise ValueError(f"Unknown task name: {task_name=}")

        # Read in the data into a Pandas DataFrame.
        # Important: The ordering `self.schema` determines the directory structure.
        self.schema = yaml.safe_load(open(self.data_dir / "schema.yaml"))
        models: dict[str, dict[str, Any]] = json.load(
            open(self.data_dir / "models.json")
        )
        res_df = pd.DataFrame()
        for choice in itertools.product(*self.schema.values()):
            result_dir = self.data_dir / "/".join(choice)
            with contextlib.suppress(FileNotFoundError):
                for model_id, model_info in models.items():
                    for file in (result_dir / model_id).glob("*.json"):
                        model_df = pd.DataFrame([json.load(open(file))])
                        # Sanity checks and standardization of schema values.
                        assert model_df["Model"].iloc[0] == model_id
                        for key, val in zip(self.schema.keys(), choice):
                            assert (
                                str(val).lower() in str(model_df[key].iloc[0]).lower()
                            )
                            model_df[key] = val
                        # Format the model name as an HTML anchor.
                        model_df["Model"] = self._wrap_model_name(model_info["url"], model_info["nickname"])
                        model_df["Total params"] = model_info["total_params"]
                        model_df["Denoising params"] = model_info["denoising_params"]
                        model_df["Resolution"] = model_info["resolution"]
                        res_df = pd.concat([res_df, model_df])

        if res_df.empty:
            raise ValueError(
                f"No benchmark JSON files were read from {self.data_dir=}."
            )

        # Order columns
        columns = res_df.columns.to_list()
        cols_to_order = ["Model", "Denoising params", "Total params"]
        cols_to_order.extend(self.schema.keys())
        columns = cols_to_order + [col for col in columns if col not in cols_to_order]
        res_df = res_df[columns]

        # Order rows
        res_df = res_df.sort_values(by=["Model", *self.schema.keys(), self.energy_col])

        self.full_df = res_df.round(2)

        # We need to set the default view separately when `gr.State` is forked.
        self.set_filter_get_df(detail_mode=False)

    def get_benchmark_checkboxes(self) -> dict[str, list[str]]:
        return self.schema

    def get_all_models(self) -> list[str]:
        return self.full_df["Model"].apply(self._unwrap_model_name).unique().tolist()

    def set_filter_get_df(self, detail_mode: bool, *filters) -> pd.DataFrame:
        """Set the current set of filters and return the filtered DataFrame.

        Filters can either be completely empty, or be a concatenated list of
        choices from all checkboxes and all sliders.
        """
        # If the filter is empty, we default to the first choice for each key.
        if not filters:
            checkboxes = [choices[:1] for choices in self.schema.values()]
            sliders = [slider[3] for slider in self.get_benchmark_sliders().values()]
            filters = checkboxes + sliders

        index = np.full(len(self.full_df), True)
        # Checkboxes
        for setup, choice in zip(self.schema, filters):
            index = index & self.full_df[setup].isin(choice)
        cur_df = self.full_df.loc[index]

        # Sliders (We just have Batch latency for now.)
        # For each `Model`, we want to first filter out rows whose `Batch latency (s)` is greater than the slider value.
        # Finally, only just leave the row whose `Energy/image (J)` or `Energy/video (J)` is the smallest.
        batch_latency = filters[-1]
        cur_df = (
            cur_df
                .groupby("Model")[cur_df.columns]
                .apply(
                    lambda x: x[x["Batch latency (s)"] <= batch_latency],
                    include_groups=True,
                )
                .sort_values(by=self.energy_col)
                .reset_index(drop=True)
                .groupby("Model")
                .head(1)
        )

        if not detail_mode:
            core_columns = ["Model", "Denoising params", "GPU", "Resolution", "Frames", self.energy_col]
            readable_name_mapping = {
                "Denoising params": "Denoising parameters (Billions)",
                "GPU": "GPU model",
                self.energy_col: self.energy_col_readable,
            }
            for column in cur_df.columns:
                if column not in core_columns:
                    cur_df = cur_df.drop(column, axis=1)
            cur_df = cur_df.rename(columns=readable_name_mapping)

        return cur_df


class DiffusionT2ITableManager(DiffusionTableManager):
    """Diffusion table manager for text-to-image tasks."""

    def get_tab_name(self) -> str:
        return "Diffusion Text to image"

    def get_intro_text(self) -> str:
        text = """
            <h2>How much energy do GenAI models consume?</h2>

            <h3>Diffusion text-to-image generation</h3>

            <p style="font-size: 16px">
            Diffusion models generate images that align with input text prompts.
            Using <a href="https://ml.energy/zeus">Zeus</a> for energy measurement, we created a leaderboard for the energy consumption of Diffusion text-to-image.
            </p>

            <p style="font-size: 16px">
            More models will be added over time. Stay tuned!
            </p>
            """
        return text

    def get_detail_text(self, detail_mode: bool) -> str:
        if detail_mode:
            text = """
                Each row corresponds to one model, given a constraint on the maximum computation time for the whole batch.
                If more than one GPU types were chosen, the row shows results from the GPU with the lowest energy consumption per image.

                Columns
                - **Model**: The name of the model.
                - **Denoising params**: Number of parameters in the denosing module (e.g., UNet, Transformer).
                - **Total params**: Total number of parameters in the model, including encoders and decoders.
                - **GPU**: Name of the GPU model used for benchmarking.
                - **Energy/image (J)**: Energy consumed per generated image in Joules.
                - **Batch latency (s)**: Time taken to generate a batch of images in seconds.
                - **Batch size**: Number of prompts/images in a batch.
                - **Denoising steps**: Number of denoising steps used for the diffusion model.
                - **Resolution**: Resolution of the generated image.

                For more detailed information, please take a look at the **About** tab.
                """
        else:
            text = """
                Columns
                - **Model**: The name of the model.
                - **Denoising parameters (Billions)**: Number of parameters in the diffusion model's (core) denoising module. This part of the model is run repetitively to generate gradually refine the image.
                - **GPU model**: Name of the GPU model used for benchmarking.
                - **Energy per image (Joules)**: Energy consumed for each generated image in Joules.
                - **Resolution**: Resolution of the generated image.

                Checking "Show more technical details" above the table will reveal more detailed columns.
                Also, for more detailed information, please take a look at the **About** tab.
                """
        return text

    def get_benchmark_sliders(self) -> dict[str, tuple[float, float, float, float]]:
        return {"Batch latency (s)": (0.0, 60.0, 1.0, 10.0)}


class DiffusionT2VTableManager(DiffusionTableManager):
    """Diffusion table manager for text-to-video tasks."""

    def get_tab_name(self) -> str:
        return "Diffusion Text to video"

    def get_intro_text(self) -> str:
        text = """
            <h2>How much energy do GenAI models consume?</h2>

            <h3>Diffusion text-to-video generation</h3>

            <p style="font-size: 16px">
            Diffusion models generate videos that align with input text prompts.
            Using <a href="https://ml.energy/zeus">Zeus</a> for energy measurement, we created a leaderboard for the energy consumption of Diffusion text-to-video.
            </p>

            <p style="font-size: 16px">
            More models will be added over time. Stay tuned!
            </p>
            """
        return text

    def get_detail_text(self, detail_mode: bool) -> str:
        if detail_mode:
            text = """
                Each row corresponds to one model, given a constraint on the maximum computation time for the whole batch.
                If more than one GPU types were chosen, the row shows results from the GPU with the lowest energy consumption per video.

                Columns
                - **Model**: The name of the model.
                - **Denoising params**: Number of parameters in the denosing module (e.g., UNet, Transformer).
                - **Total params**: Total number of parameters in the model, including encoders and decoders.
                - **GPU**: Name of the GPU model used for benchmarking.
                - **Energy/video (J)**: Energy consumed per generated video in Joules.
                - **Batch latency (s)**: Time taken to generate a batch of videos in seconds.
                - **Batch size**: Number of prompts/videos in a batch.
                - **Denoising steps**: Number of denoising steps used for the diffusion model.
                - **Frames**: Number of frames in the generated video.
                - **Resolution**: Resolution of the generated video.

                For more detailed information, please take a look at the **About** tab.
                """
        else:
            text = """
                Columns
                - **Model**: The name of the model.
                - **Denoising parameters (Billions)**: Number of parameters in the diffusion model's (core) denoising module. This part of the model is run repetitively to generate gradually refine the video.
                - **GPU model**: Name of the GPU model used for benchmarking.
                - **Energy per video (Joules)**: Energy consumed for each generated image in Joules.
                - **Frames**: Number of frames in the generated video.
                - **Resolution**: Resolution of the generated video.

                Checking "Show more technical details" above the table will reveal more detailed columns.
                Also, for more detailed information, please take a look at the **About** tab.
                """
        return text

    def get_benchmark_sliders(self) -> dict[str, tuple[float, float, float, float]]:
        return {"Batch latency (s)": (0.0, 60.0, 1.0, 10.0)}


class DiffusionI2VTableManager(DiffusionTableManager):
    """Diffusion table manager for image-to-video tasks."""

    def get_tab_name(self) -> str:
        return "Diffusion Image to video"

    def get_intro_text(self) -> str:
        text = """
            <h2>How much energy do GenAI models consume?</h2>

            <h3>Diffusion image-to-video generation</h3>

            <p style="font-size: 16px">
            Diffusion models generate videos given an input image (and sometimes alongside with text).
            Using <a href="https://ml.energy/zeus">Zeus</a> for energy measurement, we created a leaderboard for the energy consumption of Diffusion image-to-video.
            </p>

            <p style="font-size: 16px">
            More models will be added over time. Stay tuned!
            </p>
            """
        return text

    def get_detail_text(self, detail_mode: bool) -> str:
        if detail_mode:
            text = """
                Each row corresponds to one model, given a constraint on the maximum computation time for the whole batch.
                If more than one GPU types were chosen, the row shows results from the GPU with the lowest energy consumption per video.

                Columns
                - **Model**: The name of the model.
                - **Denoising params**: Number of parameters in the denosing module (e.g., UNet, Transformer).
                - **Total params**: Total number of parameters in the model, including encoders and decoders.
                - **GPU**: Name of the GPU model used for benchmarking.
                - **Energy/video (J)**: Energy consumed per generated video in Joules.
                - **Batch latency (s)**: Time taken to generate a batch of videos in seconds.
                - **Batch size**: Number of prompts/videos in a batch.
                - **Denoising steps**: Number of denoising steps used for the diffusion model.
                - **Frames**: Number of frames in the generated video.
                - **Resolution**: Resolution of the generated video.

                For more detailed information, please take a look at the **About** tab.
                """
        else:
            text = """
                Columns
                - **Model**: The name of the model.
                - **Denoising parameters (Billions)**: Number of parameters in the diffusion model's (core) denoising module. This part of the model is run repetitively to generate gradually refine the video.
                - **GPU model**: Name of the GPU model used for benchmarking.
                - **Energy per video (Joules)**: Energy consumed for each generated image in Joules.
                - **Frames**: Number of frames in the generated video.
                - **Resolution**: Resolution of the generated video.

                Checking "Show more technical details" above the table will reveal more detailed columns.
                Also, for more detailed information, please take a look at the **About** tab.
                """
        return text

    def get_benchmark_sliders(self) -> dict[str, tuple[float, float, float, float]]:
        return {"Batch latency (s)": (0.0, 120.0, 1.0, 60.0)}


class LegacyTableManager:
    def __init__(self, data_dir: str) -> None:
        """Load the legacy LLM leaderboard data from CSV files in data_dir.

        Inside `data_dir`, there should be:
        - `models.json`: a JSON file containing information about each model.
        - `schema.yaml`: a YAML file containing the schema of the benchmark.
        - `score.csv`: a CSV file containing the NLP evaluation metrics of each model.
        - `*_benchmark.csv`: CSV files containing the system benchmark results.

        Especially, the `*_benchmark.csv` files should be named after the
        parameters used in the benchmark. For example, for the CSV file that
        contains benchmarking results for A100 and the chat-concise task
        (see `schema.yaml`) for possible choices, the file should be named
        `A100_chat-concise_benchmark.csv`.
        """
        # Load and merge CSV files.
        df = self._read_tables(data_dir)

        # Add the #params column.
        models = json.load(open(f"{data_dir}/models.json"))
        df["parameters"] = df["model"].apply(lambda x: models[x]["params"])

        # Make the first column (model) an HTML anchor to the model's website.
        def format_model_link(model_name: str) -> str:
            url = models[model_name]["url"]
            nickname = models[model_name]["nickname"]
            return (
                f'<a style="text-decoration: underline; text-decoration-style: dotted" '
                f'target="_blank" href="{url}">{nickname}</a>'
            )

        df["model"] = df["model"].apply(format_model_link)

        # Sort by our 'energy efficiency' score.
        df = df.sort_values(by="energy", ascending=True)

        # The full table where all the data are.
        self.full_df = df

        # Default view of the table is to only show the first options.
        self.set_filter_get_df()

    def _read_tables(self, data_dir: str) -> pd.DataFrame:
        """Read tables."""
        df_score = pd.read_csv(f"{data_dir}/score.csv")

        with open(f"{data_dir}/schema.yaml") as file:
            self.schema: dict[str, list] = yaml.safe_load(file)

        res_df = pd.DataFrame()

        # Do a cartesian product of all the choices in the schema
        # and try to read the corresponding CSV files.
        for choice in itertools.product(*self.schema.values()):
            filepath = f"{data_dir}/{'_'.join(choice)}_benchmark.csv"
            with contextlib.suppress(FileNotFoundError):
                df = pd.read_csv(filepath)
                for key, val in zip(self.schema.keys(), choice):
                    df.insert(1, key, val)
                res_df = pd.concat([res_df, df])

        if res_df.empty:
            raise ValueError(f"No benchmark CSV files were read from {data_dir=}.")

        df = pd.merge(res_df, df_score, on=["model"]).round(2)

        # Order columns.
        columns = df.columns.to_list()
        cols_to_order = ["model"]
        cols_to_order.extend(self.schema.keys())
        cols_to_order.append("energy")
        columns = cols_to_order + [col for col in columns if col not in cols_to_order]
        df = df[columns]

        # Delete rows with *any* NaN values.
        df = df.dropna()

        return df

    def _format_msg(self, text: str) -> str:
        """Formats into HTML that prints in Monospace font."""
        return f"<pre style='font-family: monospace'>{text}</pre>"

    def get_dropdown(self):
        columns = self.full_df.columns.tolist()[1:]
        return [
            gr.Dropdown(choices=columns, value="parameters", label="X"),
            gr.Dropdown(choices=columns, value="energy", label="Y"),
            gr.Dropdown(choices=["None", *columns], label="Z (optional)"),
        ]

    def update_dropdown(self):
        columns = self.full_df.columns.tolist()[1:]
        return [
            gr.Dropdown.update(choices=columns),
            gr.Dropdown.update(choices=columns),
            gr.Dropdown.update(choices=["None", *columns]),
        ]

    def set_filter_get_df(self, *filters) -> pd.DataFrame:
        """Set the current set of filters and return the filtered DataFrame."""
        # If the filter is empty, we default to the first choice for each key.
        if not filters:
            filters = [choices[:1] for choices in self.schema.values()]

        index = np.full(len(self.full_df), True)
        for setup, choice in zip(self.schema, filters):
            index = index & self.full_df[setup].isin(choice)
        self.cur_df = self.full_df.loc[index]
        self.cur_index = index
        return self.cur_df

    def get_intro_text(self) -> str:
        """Return the leaderboard's introduction text in HTML."""
        return """
            <div align="center">
              <h2 style="color: #23d175">This is the legacy ML.ENERGY LLM leaderboard. This will be removed at the end of this year.</h2>
            </div>

            <h3>How much energy do modern Large Language Models (LLMs) consume for inference?</h3>

            <p style="font-size: 16px">
            We used <a href="https://ml.energy/zeus">Zeus</a> to benchmark various open source LLMs in terms of how much time and energy they consume for inference.
            </p>

            <p style="font-size: 16px">
            For more detailed information, please take a look at the <b>About</b> tab.
            Every benchmark is limited in some sense -- Before you interpret the results, please take a look at the <b>Limitations</b> section there, too.
            </p>
            """


# The global instance of the TableManager should only be used when
# initializing components in the Gradio interface. If the global instance
# is mutated while handling user sessions, the change will be reflected
# in every user session. Instead, the instance provided by gr.State should
# be used.
global_ltbm = LegacyTableManager("data/legacy")
global_tbms = [
    LLMChatTableManager("data/llm_text_generation/chat", "Chat"),
    LLMCodeTableManager("data/llm_text_generation/code", "Code"),
    VLMChatTableManager("data/mllm_text_generation/chat", "Visual chat"),
    DiffusionT2ITableManager("data/diffusion/text-to-image", "Text to image"),
    DiffusionT2VTableManager("data/diffusion/text-to-video", "Text to video"),
    DiffusionI2VTableManager("data/diffusion/image-to-video", "Image to video"),
]

# Custom JS.
# XXX: This is a hack to make the model names clickable.
#      Ideally, we should set `datatype` in the constructor of `gr.DataFrame` to
#      `["markdown"] + ["number"] * (len(df.columns) - 1)` and format models names
#      as an HTML <a> tag. However, because we also want to dynamically add new
#      columns to the table and Gradio < 4.0 does not support updating `datatype` with
#      `gr.DataFrame.update` yet, we need to manually walk into the DOM and replace
#      the innerHTML of the model name cells with dynamically interpreted HTML.
#      Desired feature tracked at https://github.com/gradio-app/gradio/issues/3732
dataframe_update_js = f"""
function format_model_link() {{
    // Iterate over the cells of the first column of the leaderboard table.
    var table_element = document.querySelectorAll(".tab-leaderboard");
    for (var table of table_element) {{
    for (let index = 1; index <= {len(global_ltbm.full_df) + sum(len(tbm.full_df) for tbm in global_tbms)}; index++) {{
        // Get the cell from `table`.
        var cell = table.querySelector(`div > div > div > table > tbody > tr:nth-child(${{index}}) > td:nth-child(1) > div > span`);
        // var cell = document.querySelector(
        //     `.tab-leaderboard > div > div > div > table > tbody > tr:nth-child(${{index}}) > td:nth-child(1) > div > span`
        // );

        // If nothing was found, it likely means that now the visible table has less rows
        // than the full table. This happens when the user filters the table. In this case,
        // we should just return.
        if (cell == null) break;

        // This check exists to make this function idempotent.
        // Multiple changes to the Dataframe component may invoke this function,
        // multiple times to the same HTML table (e.g., adding and sorting cols).
        // Thus, we check whether we already formatted the model names by seeing
        // whether the child of the cell is a text node. If it is not,
        // it means we already parsed it into HTML, so we should just return.
        if (cell.firstChild.nodeType != 3) break;

        // Decode and interpret the innerHTML of the cell as HTML.
        var decoded_string = new DOMParser().parseFromString(cell.innerHTML, "text/html").documentElement.textContent;
        var temp = document.createElement("template");
        temp.innerHTML = decoded_string;
        var model_anchor = temp.content.firstChild;

        // Replace the innerHTML of the cell with the interpreted HTML.
        cell.replaceChildren(model_anchor);
    }}
    }}

    // Return all arguments as is.
    return arguments
}}
"""

# Custom CSS.
custom_css = """
/* Make ML.ENERGY look like a clickable logo. */
.text-logo {
    color: #23d175 !important;
    text-decoration: none !important;
}

/* Make the submit button the same color as the logo. */
.btn-submit {
    background: #23d175 !important;
    color: white !important;
    border: 0 !important;
}

/* Center the plotly plot inside its container. */
.plotly > div {
    margin: auto !important;
}

/* Limit the width of the first column to 300 px. */
table td:first-child,
table th:first-child {
    max-width: 300px;
    overflow: auto;
    white-space: nowrap;
}

/* Make tab buttons larger */
.tab-nav > button {
    font-size: 18px !important;
}

/* Color texts. */
.green-text {
    color: #23d175 !important;
}
.red-text {
    color: #ff3860 !important;
}

/* Flashing model name borders. */
@keyframes blink {
    0%, 33%, 67%, 100% {
        border-color: transparent;
    }
    17%, 50%, 83% {
        border-color: #23d175;
    }
}
/* Older browser compatibility */
@-webkit-keyframes blink {
    0%, 33%, 67%, 100% {
        border-color: transparent;
    }
    17%, 50%, 83% {
        border-color: #23d175;
    }
}
.model-name-text {
    border: 2px solid transparent; /* Transparent border initially */
    animation: blink 3s ease-in-out 1; /* One complete cycle of animation, lasting 3 seconds */
    -webkit-animation: blink 3s ease-in-out 1; /* Older browser compatibility */
}

/* Grey out components when the Colosseum is down. */
.greyed-out {
  pointer-events: none;
  opacity: 0.4;
}

/* Make the Citation header larger */
#citation-header > div > span {
    font-size: 16px !important;
}

/* Align everything in tables to the right. */
/* Not the best solution, but at least makes the numbers align. */
.tab-leaderboard span {
    text-align: right;
}
"""

# The app will not start without a controller address set.
controller_addr = os.environ.get("COLOSSEUM_CONTROLLER_ADDR")
if controller_addr is None:
    COLOSSEUM_UP = False
    COLOSSEUM_DOWN_MESSAGE = "<br/><h2 style='text-align: center'>Local testing mode. Colosseum disabled.</h2>"
    controller_addr = "localhost"
global_controller_client = ControllerClient(controller_addr=controller_addr, timeout=15)

# Fetch the latest update date of the leaderboard repository.
resp = requests.get("https://api.github.com/repos/ml-energy/leaderboard/commits/master")
if resp.status_code != 200:
    current_date = "[Failed to fetch]"
    print("Failed to fetch the latest release date of the leaderboard repository.")
    print(resp.json())
else:
    current_datetime = parser.parse(resp.json()["commit"]["author"]["date"])
    current_date = current_datetime.astimezone(tz.gettz("US/Eastern")).strftime(
        "%Y-%m-%d"
    )

# Load the list of models. To reload, the app should be restarted.
RANDOM_MODEL_NAME = "Random"
RANDOM_USER_PREFERENCE = "Two random models"
global_available_models = global_controller_client.get_available_models() if COLOSSEUM_UP else []
model_name_to_user_pref = {model: f"One is {model}" for model in global_available_models}
model_name_to_user_pref[RANDOM_MODEL_NAME] = RANDOM_USER_PREFERENCE
user_pref_to_model_name = {v: k for k, v in model_name_to_user_pref.items()}


# Colosseum helper functions.
def enable_interact(num: int):
    def inner():
        return [gr.update(interactive=True)] * num
    return inner


def disable_interact(num: int):
    def inner():
        return [gr.update(interactive=False)] * num
    return inner


def consumed_less_energy_message(energy_a, energy_b):
    """Return a message that indicates that the user chose the model that consumed less energy.

    By default report in "%f %" but if the difference is larger than 2 times, report in "%f X".
    """
    less_energy = min(energy_a, energy_b)
    more_energy = max(energy_a, energy_b)
    factor = less_energy / more_energy
    how_much = f"{1 / factor:.1f}x" if factor <= 0.5 else f"{100 - factor * 100:.1f}%"
    return f"<h2>That response also <span class='green-text'>consumed {how_much} less energy</span> ({energy_a:,.0f} J vs. {energy_b:,.0f} J)!</h2>"


def consumed_more_energy_message(energy_a, energy_b):
    """Return a message that indicates that the user chose the model that consumed more energy.

    By default report in "%f %" but if the difference is larger than 2 times, report in "%f X".
    """
    less_energy = min(energy_a, energy_b)
    more_energy = max(energy_a, energy_b)
    factor = more_energy / less_energy
    how_much = f"{factor:.1f}x" if factor >= 2.0 else f"{factor * 100 - 100:.1f}%"
    return f"<h2>That response <span class='red-text'>consumed {how_much} more energy</span> ({energy_a:,.0f} J vs. {energy_b:,.0f} J).</h2>"


# Colosseum event handlers
def on_load():
    """Intialize the dataframe, shuffle the model preference dropdown choices."""
    dataframe = global_ltbm.set_filter_get_df()
    dataframes = [global_tbm.set_filter_get_df(detail_mode=False) for global_tbm in global_tbms]
    return dataframe, *dataframes


def add_prompt_disable_submit(prompt, history_a, history_b):
    """Add the user's prompt to the two model's history and disable further submission."""
    client = global_controller_client.fork()
    return [
        gr.Textbox.update(value=" ", interactive=False),
        gr.Button.update(interactive=False),
        history_a + [[prompt, ""]],
        history_b + [[prompt, ""]],
        client,
    ]


def generate_responses(client: ControllerClient, history_a, history_b):
    """Generate responses for the two models."""
    model_preference = RANDOM_MODEL_NAME
    for resp_a, resp_b in itertools.zip_longest(
        client.prompt(
            prompt=history_a[-1][0], index=0, model_preference=model_preference
        ),
        client.prompt(
            prompt=history_b[-1][0], index=1, model_preference=model_preference
        ),
    ):
        if resp_a is not None:
            history_a[-1][1] += resp_a
        if resp_b is not None:
            history_b[-1][1] += resp_b
        yield [history_a, history_b]


def make_resp_vote_func(victory_index: Literal[0, 1]):
    """Return a function that will be called when the user clicks on response preference vote buttons."""

    def resp_vote_func(client: ControllerClient):
        vote_response = client.response_vote(victory_index=victory_index)
        model_name_a, model_name_b = map(lambda n: f"## {n}", vote_response.model_names)
        energy_a, energy_b = vote_response.energy_consumptions
        # User liked the model that also consumed less energy.
        if (victory_index == 0 and energy_a <= energy_b) or (victory_index == 1 and energy_a >= energy_b):
            energy_message = consumed_less_energy_message(energy_a, energy_b)
            return [
                # Disable response vote buttons
                gr.Button.update(interactive=False), gr.Button.update(interactive=False),
                # Reveal model names
                gr.Markdown.update(model_name_a, visible=True), gr.Markdown.update(model_name_b, visible=True),
                # Display energy consumption comparison message
                gr.Markdown.update(energy_message, visible=True),
                # Keep energy vote buttons hidden
                gr.Button.update(visible=False, interactive=False), gr.Button.update(visible=False, interactive=False),
                # Enable reset button
                gr.Button.update(visible=True, interactive=True),
            ]
        # User liked the model that consumed more energy.
        else:
            energy_message = consumed_more_energy_message(energy_a, energy_b)
            return [
                # Disable response vote buttons
                gr.Button.update(interactive=False), gr.Button.update(interactive=False),
                # Leave model names hidden
                gr.Markdown.update(visible=False), gr.Markdown.update(visible=False),
                # Display energy consumption comparison message
                gr.Markdown.update(energy_message, visible=True),
                # Reveal and enable energy vote buttons
                gr.Button.update(visible=True, interactive=True), gr.Button.update(visible=True, interactive=True),
                # Keep the reset button disabled
                gr.Button.update(visible=False, interactive=False),
            ]

    return resp_vote_func


def make_energy_vote_func(is_worth: bool):
    """Return a function that will be called when the user clicks on energy vote buttons."""

    def energy_vote_func(client: ControllerClient, energy_message: str):
        vote_response = client.energy_vote(is_worth=is_worth)
        model_name_a, model_name_b = map(lambda n: f"## {n}", vote_response.model_names)
        return [
            # Reveal model names
            gr.Markdown.update(model_name_a, visible=True), gr.Markdown.update(model_name_b, visible=True),
            # Disable energy vote buttons
            gr.Button.update(interactive=False), gr.Button.update(interactive=False),
            # Enable reset button
            gr.Button.update(interactive=True, visible=True),
            # Append to the energy comparison message
            energy_message[:-5] + (" Fair enough.</h2>" if is_worth else " Wasn't worth it.</h2>"),
        ]

    return energy_vote_func


def play_again():
    available_models = copy.deepcopy(global_available_models)
    random.shuffle(available_models)
    available_models.insert(0, RANDOM_MODEL_NAME)
    return [
        # Clear chatbot history
        None, None,
        # Enable prompt textbox and submit button
        gr.Textbox.update(value="", interactive=True), gr.Button.update(interactive=True),
        # Mask model names
        gr.Markdown.update(value="", visible=False), gr.Markdown.update(value="", visible=False),
        # Hide energy vote buttons and message
        gr.Button.update(visible=False), gr.Button.update(visible=False), gr.Markdown.update(visible=False),
        # Disable reset button
        gr.Button.update(interactive=False, visible=False),
    ]


def toggle_detail_mode_slider_visibility(detail_mode: bool, *sliders):
    return [detail_mode] + [gr.update(visible=detail_mode)] * len(sliders)


def toggle_detail_mode_sync_tabs(detail_mode: bool, *checkboxes):
    return [gr.Checkbox.update(value=detail_mode)] * len(checkboxes) + [gr.Markdown.update(tbm.get_detail_text(detail_mode)) for tbm in global_tbms]


focus_prompt_input_js = """
function() {
    for (let textarea of document.getElementsByTagName("textarea")) {
        if (textarea.hasAttribute("autofocus")) {
            textarea.focus();
            return;
        }
    }
}
"""

with gr.Blocks(css=custom_css) as block:
    tbm = gr.State(global_ltbm)  # type: ignore
    local_tbms: list[TableManager] = [gr.State(global_tbm) for global_tbm in global_tbms]  # type: ignore
    detail_mode = gr.State(False)  # type: ignore

    with gr.Box():
        gr.HTML(
            "<h1><a href='https://ml.energy' class='text-logo'>ML.ENERGY</a> Leaderboard</h1>"
        )

    with gr.Tabs():
        # Tab: Colosseum.
        with gr.Tab("Colosseum ⚔️️"):
            if COLOSSEUM_UP:
                gr.Markdown(open("docs/colosseum_top.md").read())
            else:
                gr.HTML(COLOSSEUM_DOWN_MESSAGE)
                gr.HTML("<h3 style='text-align: center'>The energy leaderboard is still available.</h3><br/>")
                gr.HTML(COLOSSUMM_YOUTUBE_DEMO_EMBED_HTML)

            with gr.Group():
                with gr.Row():
                    prompt_input = gr.Textbox(
                        show_label=False,
                        placeholder="Input your prompt, e.g., 'Explain machine learning in simple terms.'",
                        container=False,
                        scale=20,
                        interactive=COLOSSEUM_UP,
                        elem_classes=None if COLOSSEUM_UP else ["greyed-out"],
                    )
                    prompt_submit_btn = gr.Button(
                        value="⚔️️ Fight!",
                        elem_classes=["btn-submit"] if COLOSSEUM_UP else ["greyed-out"],
                        min_width=60,
                        scale=1,
                        interactive=COLOSSEUM_UP,
                    )

            with gr.Row():
                masked_model_names = []
                chatbots = []
                resp_vote_btn_list: list[gr.component.Component] = []
                with gr.Column():
                    with gr.Row():
                        masked_model_names.append(
                            gr.Markdown(visible=False, elem_classes=["model-name-text"])
                        )
                    with gr.Row():
                        chatbots.append(
                            gr.Chatbot(
                                label="Model A",
                                elem_id="chatbot",
                                height=400,
                                elem_classes=None if COLOSSEUM_UP else ["greyed-out"],
                            )
                        )
                    with gr.Row():
                        left_resp_vote_btn = gr.Button(
                            value="👈 Model A is better", interactive=False
                        )
                        resp_vote_btn_list.append(left_resp_vote_btn)

                with gr.Column():
                    with gr.Row():
                        masked_model_names.append(
                            gr.Markdown(visible=False, elem_classes=["model-name-text"])
                        )
                    with gr.Row():
                        chatbots.append(
                            gr.Chatbot(
                                label="Model B",
                                elem_id="chatbot",
                                height=400,
                                elem_classes=None if COLOSSEUM_UP else ["greyed-out"],
                            )
                        )
                    with gr.Row():
                        right_resp_vote_btn = gr.Button(
                            value="👉 Model B is better", interactive=False
                        )
                        resp_vote_btn_list.append(right_resp_vote_btn)

            with gr.Row():
                energy_comparison_message = gr.HTML(visible=False)

            with gr.Row():
                worth_energy_vote_btn = gr.Button(
                    value="The better response was worth 👍 the extra energy.",
                    visible=False,
                )
                notworth_energy_vote_btn = gr.Button(
                    value="Not really worth that much more. 👎", visible=False
                )
                energy_vote_btn_list: list[gr.component.Component] = [
                    worth_energy_vote_btn,
                    notworth_energy_vote_btn,
                ]

            with gr.Row():
                play_again_btn = gr.Button(
                    "Play again!", visible=False, elem_classes=["btn-submit"]
                )

            gr.Markdown(open("docs/colosseum_bottom.md").read())

            controller_client = gr.State()


            (prompt_input
                .submit(add_prompt_disable_submit, [prompt_input, *chatbots], [prompt_input, prompt_submit_btn, *chatbots, controller_client], queue=False)
                .then(generate_responses, [controller_client, *chatbots], [*chatbots], queue=True, show_progress="hidden")
                .then(enable_interact(2), None, resp_vote_btn_list, queue=False))
            (prompt_submit_btn
                .click(add_prompt_disable_submit, [prompt_input, *chatbots], [prompt_input, prompt_submit_btn, *chatbots, controller_client], queue=False)
                .then(generate_responses, [controller_client, *chatbots], [*chatbots], queue=True, show_progress="hidden")
                .then(enable_interact(2), None, resp_vote_btn_list, queue=False))

            left_resp_vote_btn.click(
                make_resp_vote_func(victory_index=0),
                [controller_client],
                [*resp_vote_btn_list, *masked_model_names, energy_comparison_message, *energy_vote_btn_list, play_again_btn],
                queue=False,
            )
            right_resp_vote_btn.click(
                make_resp_vote_func(victory_index=1),
                [controller_client],
                [*resp_vote_btn_list, *masked_model_names, energy_comparison_message, *energy_vote_btn_list, play_again_btn],
                queue=False,
            )

            worth_energy_vote_btn.click(
                make_energy_vote_func(is_worth=True),
                [controller_client, energy_comparison_message],
                [*masked_model_names, *energy_vote_btn_list, play_again_btn, energy_comparison_message],
                queue=False,
            )
            notworth_energy_vote_btn.click(
                make_energy_vote_func(is_worth=False),
                [controller_client, energy_comparison_message],
                [*masked_model_names, *energy_vote_btn_list, play_again_btn, energy_comparison_message],
                queue=False,
            )

            (play_again_btn
                .click(
                    play_again,
                    None,
                    [*chatbots, prompt_input, prompt_submit_btn, *masked_model_names, *energy_vote_btn_list, energy_comparison_message, play_again_btn],
                    queue=False,
                )
                .then(None, _js=focus_prompt_input_js, queue=False))

        # Tab: Leaderboards.
        dataframes = []
        all_detail_mode_checkboxes = []
        all_sliders = []
        all_detail_text_components = []
        for global_tbm, local_tbm in zip(global_tbms, local_tbms):
            with gr.Tab(global_tbm.get_tab_name()):
                # Box: Introduction text.
                with gr.Box():
                    gr.Markdown(global_tbm.get_intro_text())

                # Block: Checkboxes and sliders to select benchmarking parameters. A detail mode checkbox.
                with gr.Row():
                    checkboxes: list[gr.CheckboxGroup] = []
                    for key, choices in global_tbm.get_benchmark_checkboxes().items():
                        # Check the first element by default.
                        checkboxes.append(gr.CheckboxGroup(choices=choices, value=choices[:1], label=key))

                    sliders: list[gr.Slider] = []
                    for key, (min_val, max_val, step, default) in global_tbm.get_benchmark_sliders().items():
                        sliders.append(gr.Slider(minimum=min_val, maximum=max_val, value=default, step=step, label=key, visible=detail_mode.value))
                    all_sliders.extend(sliders)

                with gr.Row():
                    detail_mode_checkbox = gr.Checkbox(label="Show more technical details", value=False)
                    all_detail_mode_checkboxes.append(detail_mode_checkbox)

                # Block: Leaderboard table.
                with gr.Row():
                    dataframe = gr.Dataframe(
                        type="pandas",
                        elem_classes=["tab-leaderboard"],
                        interactive=False,
                        max_rows=1000,
                    )
                    dataframes.append(dataframe)

                    # Make sure the models have clickable links.
                    dataframe.change(
                        None, None, None, _js=dataframe_update_js, queue=False
                    )
                    # Table automatically updates when users check or uncheck any checkbox or move any slider.
                    for element in [detail_mode_checkbox, *checkboxes, *sliders]:
                        element.change(
                            global_tbm.__class__.set_filter_get_df,
                            inputs=[local_tbm, detail_mode, *checkboxes, *sliders],
                            outputs=dataframe,
                            queue=False,
                        )

                # Block: More details about the leaderboard.
                with gr.Box():
                    detail_text = global_tbm.get_detail_text(detail_mode=False)
                    all_detail_text_components.append(gr.Markdown(detail_text))

                # Block: Leaderboard date.
                with gr.Row():
                    gr.HTML(
                        f"<h3 style='color: gray'>Last updated: {current_date}</h3>"
                    )

        # Tab: Legacy leaderboard.
        with gr.Tab("LLM Leaderboard (legacy)"):
            with gr.Box():
                gr.Markdown(global_ltbm.get_intro_text())

            # Block: Checkboxes to select benchmarking parameters.
            with gr.Row():
                with gr.Box():
                    gr.Markdown("### Benchmark results to show")
                    checkboxes: list[gr.CheckboxGroup] = []
                    for key, choices in global_ltbm.schema.items():
                        # Specifying `value` makes everything checked by default.
                        checkboxes.append(
                            gr.CheckboxGroup(
                                choices=choices, value=choices[:1], label=key
                            )
                        )

            # Block: Leaderboard table.
            with gr.Row():
                dataframe = gr.Dataframe(
                    type="pandas", elem_classes=["tab-leaderboard"], interactive=False
                )
            # Make sure the models have clickable links.
            dataframe.change(None, None, None, _js=dataframe_update_js, queue=False)
            # Table automatically updates when users check or uncheck any checkbox.
            for checkbox in checkboxes:
                checkbox.change(
                    LegacyTableManager.set_filter_get_df,
                    inputs=[tbm, *checkboxes],
                    outputs=dataframe,
                    queue=False,
                )

            # Block: Leaderboard date.
            with gr.Row():
                gr.HTML(f"<h3 style='color: gray'>Last updated: {current_date}</h3>")

        # Tab: About page.
        with gr.Tab("About"):
            gr.Markdown(open("docs/about.md").read())

        # Detail mode toggling.
        for detail_mode_checkbox in all_detail_mode_checkboxes:
            detail_mode_checkbox.change(
                toggle_detail_mode_slider_visibility,
                inputs=[detail_mode_checkbox, *all_sliders],
                outputs=[detail_mode, *all_sliders],
                queue=False,
            )
            detail_mode_checkbox.change(
                toggle_detail_mode_sync_tabs,
                inputs=[detail_mode_checkbox, *all_detail_mode_checkboxes],
                outputs=[*all_detail_mode_checkboxes, *all_detail_text_components],
                queue=False,
            )

    # Citation
    with gr.Accordion("📚  Citation", open=False, elem_id="citation-header"):
        citation_text = open("docs/citation.bib").read()
        gr.Textbox(
            value=citation_text,
            label="BibTeX for the leaderboard and the Zeus framework used for benchmarking:",
            lines=len(list(filter(lambda c: c == "\n", citation_text))),
            interactive=False,
            show_copy_button=True,
        )

    # Load the table on page load.
    block.load(
        on_load,
        outputs=[dataframe, *dataframes],
        queue=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--share", action="store_true", help="Specify if sharing is enabled"
    )
    parser.add_argument("--concurrency", type=int, default=50)

    args = parser.parse_args()
    block.queue(concurrency_count=args.concurrency, api_open=False).launch(
        share=args.share, show_error=True
    )
