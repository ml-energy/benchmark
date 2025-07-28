"""
Modified based on:
https://github.com/vllm-project/vllm/blob/v0.10.0/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py

required packages:
    quart
"""

import socket
import threading
import time
import uuid
from typing import Any

import aiohttp
import asyncio
import msgpack
import zmq
from quart import Quart, make_response, request
import argparse

count = 0
prefill_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)
decode_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)

prefill_cv = threading.Condition()
decode_cv = threading.Condition()

DEFAULT_PING_SECONDS = 5

parser = argparse.ArgumentParser()
parser.add_argument(
    "--port",
    type=int,
    default=10001,
    help="Port to bind the proxy server to (default: 10001)",
)
parser.add_argument(
    "--discovery-port",
    type=int,
    default=30001,
    help="Port for service discovery (default: 30001)",
)
parser.add_argument(
    "--num-prefills",
    type=int,
    required=False,
    help="Number of prefill instances to register",
)
parser.add_argument(
    "--num-decodes",
    type=int,
    required=False,
    help="Number of decode instances to register",
)
args = parser.parse_args()

def _remove_oldest_instances(instances: dict[str, Any]) -> None:
    oldest_key = next(iter(instances), None)
    while oldest_key is not None:
        value = instances[oldest_key]
        if value[1] > time.time():
            break
        print(f"🔴Remove [HTTP:{oldest_key}, ZMQ:{value[0]}, stamp:{value[1]}]")
        instances.pop(oldest_key, None)
        oldest_key = next(iter(instances), None)


def _listen_for_register(poller, router_socket):
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_address, message = router_socket.recv_multipart()
            # data: {"type": "P", "http_address": "ip:port",
            #        "zmq_address": "ip:port"}
            data = msgpack.loads(message)
            if data["type"] == "P":
                global prefill_instances
                global prefill_cv
                with prefill_cv:
                    node = prefill_instances.pop(data["http_address"], None)
                    prefill_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(prefill_instances)

            elif data["type"] == "D":
                global decode_instances
                global decode_cv
                with decode_cv:
                    node = decode_instances.pop(data["http_address"], None)
                    decode_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(decode_instances)
            else:
                print(
                    "Unexpected, Received message from %s, data: %s",
                    remote_address,
                    data,
                )

            if node is None:
                print(f"🔵Add [HTTP:{data['http_address']}, ZMQ:{data['zmq_address']}]")


def start_service_discovery(hostname, port):
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")

    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    _listener_thread = threading.Thread(
        target=_listen_for_register, args=[poller, router_socket], daemon=True
    )
    _listener_thread.start()
    return _listener_thread


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)
app.config.update(
    RESPONSE_TIMEOUT = 3600,
    BODY_TIMEOUT     = 3600,
)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


async def forward_request(url, data, request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "X-Request-Id": request_id,
        }
        try:
            async with session.post(url=url, json=data, headers=headers) as response:
                if response.status == 200:
                    # async for chunk_bytes in response.content.iter_chunked(1024):
                    #     yield chunk_bytes
                    async for chunk, _ in response.content.iter_chunks():
                        yield chunk
                else:
                    print(f"Error forwarding request to {url}: {response.status}")
                    raise Exception(
                        f"Error forwarding request to {url}: {response.status}"
                    )
        except Exception as e:
            print(f"Error forwarding request to {url}: {e}")
            raise e


@app.route("/v1/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request["max_tokens"] = 1

        global count
        global prefill_instances
        global prefill_cv
        with prefill_cv:
            prefill_list = list(prefill_instances.items())
            prefill_addr, prefill_zmq_addr = prefill_list[count % len(prefill_list)]
            prefill_zmq_addr = prefill_zmq_addr[0]

        global decode_instances
        global decode_cv
        with decode_cv:
            decode_list = list(decode_instances.items())
            decode_addr, decode_zmq_addr = decode_list[count % len(decode_list)]
            decode_zmq_addr = decode_zmq_addr[0]

        print(
            f"handle_request count: {count}, [HTTP:{prefill_addr}, "
            f"ZMQ:{prefill_zmq_addr}] 👉 [HTTP:{decode_addr}, "
            f"ZMQ:{decode_zmq_addr}]"
        )
        count += 1

        request_id = (
            f"___prefill_addr_{prefill_zmq_addr}___decode_addr_"
            f"{decode_zmq_addr}_{random_uuid()}"
        )

        async def prefill_decode_gen():
            # finish prefill
            async for chunk in forward_request(
                f"http://{prefill_addr}/v1/completions", prefill_request, request_id
            ):
                print("Prefill chunk received:", chunk)
                # here we need to skip `data: [DONE]` chunk
                if b"data: [DONE]" in chunk.replace(b"\r", b"").strip():
                    continue
                yield chunk
                # continue

            # return decode
            async for chunk in forward_request(
                f"http://{decode_addr}/v1/completions", original_request_data, request_id
            ):
                yield chunk
        generator = prefill_decode_gen()
        response = await make_response(generator)
        response.timeout = None

        return response

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        return await make_response(
            "An error occurred while processing the request.", 500
        )


@app.route("/v1/chat/completions", methods=["POST"])
async def handle_chat_request():
    try:
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request["max_tokens"] = 1

        global count
        global prefill_instances
        global prefill_cv
        with prefill_cv:
            prefill_list = list(prefill_instances.items())
            prefill_addr, prefill_zmq_addr = prefill_list[count % len(prefill_list)]
            prefill_zmq_addr = prefill_zmq_addr[0]

        global decode_instances
        global decode_cv
        with decode_cv:
            decode_list = list(decode_instances.items())
            decode_addr, decode_zmq_addr = decode_list[count % len(decode_list)]
            decode_zmq_addr = decode_zmq_addr[0]

        print(
            f"handle_request count: {count}, [HTTP:{prefill_addr}, "
            f"ZMQ:{prefill_zmq_addr}] 👉 [HTTP:{decode_addr}, "
            f"ZMQ:{decode_zmq_addr}]"
        )
        count += 1

        request_id = (
            f"___prefill_addr_{prefill_zmq_addr}___decode_addr_"
            f"{decode_zmq_addr}_{random_uuid()}"
        )

        async def prefill_decode_gen():
            # finish prefill
            async for chunk in forward_request(
                f"http://{prefill_addr}/v1/completions", prefill_request, request_id
            ):
                print("Prefill chunk received:", chunk)
                # here we need to skip `data: [DONE]` chunk
                if b"data: [DONE]" in chunk.replace(b"\r", b"").strip():
                    continue
                yield chunk

            # return decode
            async for chunk in forward_request(
                f"http://{decode_addr}/v1/completions", original_request_data, request_id
            ):
                yield chunk
        generator = prefill_decode_gen()
        response = await make_response(generator)
        response.timeout = None

        return response

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        return await make_response(
            "An error occurred while processing the request.", 500
        )

@app.route("/health", methods=["GET"])
async def health_check():
    """
    Health check endpoint to verify all instances are alive.
    """
    if args.num_prefills is not None and len(prefill_instances) < args.num_prefills:
        return await make_response(
            f"Not enough prefill instances registered: {len(prefill_instances)} < {args.num_prefills}",
            503,
        )
    if args.num_decodes is not None and len(decode_instances) < args.num_decodes:
        return await make_response(
            f"Not enough decode instances registered: {len(decode_instances)} < {args.num_decodes}",
            503,
        )
    coros = []
    async def instance_healthy(http_address: str) -> bool:
        try:
            async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
                async with session.get(f"http://{http_address}/health") as response:
                    if response.status == 200:
                        print("Instance healthy:", http_address)
                        return True
                    return False
        except Exception as e:
            print(f"Health check failed for {http_address}: {e}")
            return False
    
    for http_address in prefill_instances.keys():
        coros.append(instance_healthy(http_address))
    for http_address in decode_instances.keys():
        coros.append(instance_healthy(http_address))
    results = await asyncio.gather(*coros)
    if all(results):
        return await make_response("All instances are healthy", 200)
    else:
        return await make_response("Some instances are unhealthy", 503)

if __name__ == "__main__":
    t = start_service_discovery("0.0.0.0", args.discovery_port)
    app.run(host="0.0.0.0", port=args.port)
    t.join()
