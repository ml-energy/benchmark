from __future__ import annotations

from spitfight.colosseum.client import ControllerClient


def test_new_uuid_on_deepcopy():
    client = ControllerClient("http://localhost:8000")
    clients = [client.fork() for _ in range(50)]
    request_ids = [client.request_id for client in clients]
    assert len(set(request_ids)) == len(request_ids)
