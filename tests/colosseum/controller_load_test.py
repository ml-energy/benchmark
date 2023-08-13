import os
import time
import random
import itertools
import multiprocessing as mp

import tyro

from spitfight.colosseum.client import ControllerClient

CONTROLLER_ADDR = os.environ["COLOSSEUM_CONTROLLER_ADDR"]

PROMPTS = [
    "What is Deep Learning?",
    "Write a poem about life.",
    "What is the basics of Rust?",
    "What is Python's GIL?",
    "What are Go channels and how do they compare with Rust flume channels?",
    "What is the difference between a list and a tuple in Python?",
    "How do I use Python's asyncio.wait?",
    "How do I accurately measure the execution time of a function in Python?",
    "How do I use Python's multiprocessing module?",
    "What is Python's built-in dataclasses module?",
    "How is Python's async/await different from Rust's async/await?",
    "What is Hugging Face Transformers?",
    "Tell me about your capabilities.",
    "When is your knowledge cutoff, and what does it mean?",
    "Explain Machine Learning in simple terms.",
    "Write a song that welcomes new students to the University of Michigan.",
    "Explain how to use the Pydantic library with a single code block.",
    "Write a poem about Jae-Won Chung, God of Computer Science.",
    "Write a poem about the University of Michigan.",
    "How do I get my new AI startup funded?",
    "Explain the notion of zero copy in programming.",
    "Explain the notion of zero knowledge proofs.",
    "Explain the notion of zero trust in cybersecurity.",
    "What is a monad in functional programming?",
    "What is a monad in category theory?",
    "How are monads implemented in both Haskell and OCaml?",
    "What is the difference between a monad and a functor?",
    "What is the difference between a monad and a monoid?",
    "How are monads used in Rust?",
    "What is a good name for a software library that makes ML energy efficient?",
    "What would be some good naming criteria for a tech startup?",
    "What is the opposite of democracy? Explain in detail.",
    "Why are people scared to be contacted by the IRS?",
    "What is fingerstyle guitar?",
    "How do I practice and play fingerstyle guitar?",
    "What is the difference between fingerstyle and classical guitar?",
    "What is the difference between classical and flamenco guitar?",
    "What is the difference between classical and jazz guitar?",
    "Explain the basics of the Django web framework.",
    "Explain the basics of the Flask web framework.",
    "Explain the basics of the FastAPI web framework.",
    "I really need to pee. What should I do?",
    "Why would one use Python's abc module?",
    "Explain Python type annotations and why they are useful.",
    "How do I create an immutable list in Python?",
    "How do I create a mutable tuple in Python?",
    "When does dropping out of a Computer Science PhD program make sense?",
    "What is the difference between a PhD and a Masters in Computer Science?",
    "How are software engineers and software developers different?",
    "Hi",
    "What's up",
    "How are you?",
    "What am I supposed to type here",
    "Is indoor vaping legal?",
    "What are the key points of the 14th amendment?",
    "I'm new to the US. What are some social taboos I should be aware of?",
] * 2


def request(prompt: str) -> tuple[str, str, str, str, float]:
    time.sleep(random.random() * 5)
    client = ControllerClient(CONTROLLER_ADDR, timeout=30)
    
    response_a, response_b = "", ""
    start_time = time.monotonic()
    for resp_a, resp_b in itertools.zip_longest(
        client.prompt(prompt, index=0),
        client.prompt(prompt, index=1),
    ):
        if resp_a is not None:
            response_a += resp_a
        if resp_b is not None:
            response_b += resp_b

    latency = time.monotonic() - start_time
    return client.request_id, prompt, response_a, response_b, latency


def main(concurrency: int = len(PROMPTS), logfile: str = "load_test_results.csv"):
    latencies = []

    start_time = time.monotonic()
    with mp.Pool(processes=concurrency) as pool:
        for request_id, prompt, response_a, response_b, latency in pool.imap_unordered(request, PROMPTS):
            latencies.append(latency)
            print(f"Request ID {request_id} finished, {latency=:.2f}s")

    total_time = time.monotonic() - start_time
    average_latency = sum(latencies) / len(latencies)
    requests_per_second = len(latencies) / total_time
    print(f"Total time: {total_time:.2f}s")
    print(f"Average latency: {average_latency:.2f}s")
    print(f"Requests per second: {requests_per_second:.2f}")
    with open(logfile, "a") as f:
        f.write(f"{concurrency},{total_time},{average_latency},{requests_per_second}\n")


if __name__ == "__main__":
    tyro.cli(main)
