### Technical details

- We allow models to generate only up to 512 new tokens. Due to this, some responses may be cut off in the middle.
- Tokens are sampled from the model output with `temperature` 1.0, `repetition_penalty` 1.0, `top_k` 50, and `top_p` 0.95.
- Large models (>= 30B) run on two NVIDIA A40 GPUs with tensor parallelism, whereas other models run on one NVIDIA A40 GPU. We directly measure the energy consumption of these GPUs.

### Contact

Please direct general questions and issues related to the Colosseum to our GitHub repository's [discussion board](https://github.com/ml-energy/leaderboard/discussions).
You can find the ML.ENERGY initiative members in [our homepage](https://ml.energy#members).
If you need direct communication, please email admins@ml.energy.
