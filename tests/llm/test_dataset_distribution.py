"""Tests for LLM datasets and their distributions."""

from __future__ import annotations

import pytest
import numpy as np
from scipy import stats
from transformers import AutoTokenizer

from mlenergy.llm.datasets import ParetoExpDistributionDataset


class TestParetoExpDistributionDataset:
    """Test that ParetoExpDistributionDataset generates correct length distributions."""

    @pytest.fixture
    def tokenizer(self):
        """Load a test tokenizer."""
        return AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    @pytest.fixture
    def dataset(self, tokenizer, random_seed, input_mean, output_mean):
        """Create a test dataset with known parameters."""
        return ParetoExpDistributionDataset(
            random_seed=random_seed,
            input_mean=input_mean,
            output_mean=output_mean,
            model_max_length=tokenizer.model_max_length,
        )

    @pytest.mark.parametrize("random_seed", [42, 48105])
    @pytest.mark.parametrize("input_mean", [500, 1000, 2000])
    @pytest.mark.parametrize("output_mean", [500, 1000, 2000])
    @pytest.mark.parametrize("num_samples", [1000, 2500, 4000])
    def test_input_output_length_distribution(
        self, dataset, tokenizer, random_seed, input_mean, output_mean, num_samples
    ):
        """Test that input lengths follow Pareto distribution."""
        requests = dataset.sample(tokenizer=tokenizer, num_requests=num_samples)

        # Test input length distribution
        input_lengths = [request.prompt_len for request in requests]
        input_lengths = np.array(input_lengths)

        # Basic statistics checks
        assert len(input_lengths) == num_samples
        assert np.all(input_lengths > 0), "All input lengths should be positive"
        assert np.all(input_lengths <= dataset.max_length), (
            "All input lengths should be <= max_length"
        )

        # Check mean is approximately correct (within 20% tolerance)
        actual_mean = np.mean(input_lengths)
        expected_mean = dataset.input_mean
        assert abs(actual_mean - expected_mean) / expected_mean < 0.1, (
            f"Input mean {actual_mean} too far from expected {expected_mean}"
        )

        # Kolmogorov-Smirnov test for Pareto distribution
        pareto_b = dataset.input_mean * (dataset.pareto_a - 1) / dataset.pareto_a
        D, p_value = stats.kstest(
            input_lengths, "pareto", args=(dataset.pareto_a, 0, pareto_b)
        )
        assert p_value > 0.01, f"KS test failed for Pareto (p={p_value:.3f})"

        # Test output length distribution
        output_lengths = [request.expected_output_len for request in requests]
        output_lengths = np.array(output_lengths)

        # Basic statistics checks
        assert len(output_lengths) == num_samples
        assert np.all(output_lengths > 0), "All output lengths should be positive"
        assert np.all(output_lengths <= dataset.max_length), (
            "All output lengths should be <= max_length"
        )

        # Check mean is approximately correct (within 20% tolerance)
        actual_mean = np.mean(output_lengths)
        expected_mean = dataset.output_mean
        assert abs(actual_mean - expected_mean) / expected_mean < 0.1, (
            f"Output mean {actual_mean} too far from expected {expected_mean}"
        )

        # Kolmogorov-Smirnov test for Exponential distribution
        loc, scale = 0, dataset.output_mean
        D, p_value = stats.kstest(output_lengths, "expon", args=(loc, scale))
        assert p_value > 0.01, f"KS test failed for Exponential (p={p_value:.3f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
