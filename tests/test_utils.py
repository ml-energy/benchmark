from __future__ import annotations

from spitfight.utils import TokenGenerationBuffer


def test_basic1():
    buffer = TokenGenerationBuffer(stop_str="stop")

    buffer.append("hello")
    assert buffer.pop() == "hello"
    assert buffer.pop() == None
    assert not buffer.matched_stop_str

    buffer.append("world")
    assert buffer.pop() == "world"
    assert not buffer.matched_stop_str

    buffer.append("stop")
    assert buffer.pop() == None
    assert buffer.matched_stop_str
    assert buffer.pop() == None
    assert buffer.matched_stop_str
    assert buffer.pop() == None
    assert buffer.matched_stop_str
    assert buffer.pop() == None
    assert buffer.matched_stop_str

def test_basic2():
    buffer = TokenGenerationBuffer(stop_str="stop")

    buffer.append("hi")
    assert buffer.pop() == "hi"
    assert not buffer.matched_stop_str

    buffer.append("stole")
    assert buffer.pop() == "stole"
    assert not buffer.matched_stop_str

    buffer.append("sto")
    assert buffer.pop() == None
    assert not buffer.matched_stop_str

    buffer.append("ic")
    assert buffer.pop() == "stoic"
    assert not buffer.matched_stop_str

    buffer.append("st")
    assert buffer.pop() == None
    assert not buffer.matched_stop_str

    buffer.append("opper")
    assert buffer.pop() == "stopper"
    assert not buffer.matched_stop_str

    buffer.append("sto")
    assert buffer.pop() == None
    assert not buffer.matched_stop_str

    buffer.append("p")
    assert buffer.pop() == None
    assert buffer.matched_stop_str

def test_falcon1():
    buffer = TokenGenerationBuffer(stop_str="\nUser")

    buffer.append("Hi")
    assert buffer.pop() == "Hi"
    assert not buffer.matched_stop_str

    buffer.append("!")
    assert buffer.pop() == "!"
    assert not buffer.matched_stop_str

    buffer.append("\n")
    assert buffer.pop() == None
    assert not buffer.matched_stop_str

    buffer.append("User")
    assert buffer.pop() == None
    assert buffer.matched_stop_str

def test_falcon2():
    buffer = TokenGenerationBuffer(stop_str="\nUser")

    buffer.append("\n")
    assert buffer.pop() == None
    assert not buffer.matched_stop_str

    buffer.append("\n")
    assert buffer.pop() == "\n"
    assert not buffer.matched_stop_str

    buffer.append("\n")
    assert buffer.pop() == "\n"
    assert not buffer.matched_stop_str

    buffer.append("\n")
    assert buffer.pop() == "\n"
    assert not buffer.matched_stop_str

    buffer.append("User")
    assert buffer.pop() == None
    assert buffer.pop() == None
    assert buffer.matched_stop_str

def test_no_stop_str():
    buffer = TokenGenerationBuffer(stop_str=None)

    buffer.append("hello")
    assert buffer.pop() == "hello"
    assert buffer.pop() == None
    assert not buffer.matched_stop_str

    buffer.append("world")
    assert buffer.pop() == "world"
    assert buffer.pop() == None
    assert not buffer.matched_stop_str

    buffer.append("\n")
    assert buffer.pop() == "\n"
    assert buffer.pop() == None
    assert not buffer.matched_stop_str
