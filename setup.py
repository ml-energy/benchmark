from setuptools import setup, find_packages

extras_require = {
    "colosseum-controller": [
        "fastapi",
        "fschat==0.2.23",
        "text_generation @ git+https://github.com/ml-energy/text_generation_energy@master",
    ],
    "app": [
        "plotly==5.15.0",
        "gradio==3.39.0",
        "text_generation @ git+https://github.com/ml-energy/text_generation_energy@master",
    ],
    "benchmark": [
        "zeus-ml",
        "fschat==0.2.23",
        "torch==2.0.1",
        "tyro",
        "rich",
        "datasets==2.15.0",
        "diffusers==0.23.1",
        "transformers==4.35.2",
        "accelerat==0.24.1",
        "torchmetrics==1.2.0",
        "pillow==10.1.0",
    ],
    "dev": ["pytest"],
}

extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name="spitfight",
    version="0.0.1",
    url="https://github.com/ml-energy/leaderboard",
    packages=find_packages("."),
    extras_require=extras_require,
)
