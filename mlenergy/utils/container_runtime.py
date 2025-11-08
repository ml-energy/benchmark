"""Abstract container runtime interface for Docker and Singularity."""

from __future__ import annotations

import subprocess
import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)


class CleanupHandle(ABC):
    """Abstract base class for cleanup handles."""

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up the container/process."""
        pass


class DockerCleanupHandle(CleanupHandle):
    """Cleanup handle for Docker containers."""

    def __init__(self, container_name: str) -> None:
        self.container_name = container_name

    def cleanup(self) -> None:
        logger.info(f"Removing Docker container: {self.container_name}")
        subprocess.run(["docker", "rm", "-f", self.container_name], check=False)
        logger.info(f"Docker container {self.container_name} removed.")


class SingularityCleanupHandle(CleanupHandle):
    """Cleanup handle for Singularity processes."""

    def __init__(self, process_handle: subprocess.Popen) -> None:
        self.process_handle = process_handle

    def cleanup(self) -> None:
        logger.info("Terminating Singularity exec process.")
        try:
            self.process_handle.terminate()
            self.process_handle.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process_handle.kill()
            self.process_handle.wait()
        logger.info("Singularity exec process terminated.")


CleanupHandleT = TypeVar("CleanupHandleT", bound=CleanupHandle)


class ContainerRuntime(ABC, Generic[CleanupHandleT]):
    """Abstract base class for container runtimes."""

    @abstractmethod
    def build_run_command(
        self,
        image: str,
        container_name: str,
        gpu_ids: list[int],
        env_vars: dict[str, str],
        bind_mounts: list[tuple[str, str, str]],
        command: list[str],
    ) -> list[str]:
        """Build the container run command.

        Args:
            image: Container image (Docker image name or .sif file path)
            container_name: Name for the container instance
            gpu_ids: List of GPU IDs to use
            env_vars: Environment variables to set
            bind_mounts: List of (host_path, container_path, mode) tuples
            command: Command to run inside the container

        Returns:
            Command as list of strings
        """
        pass

    @abstractmethod
    def get_cleanup_handle(
        self, container_name: str, process_handle: subprocess.Popen | None
    ) -> CleanupHandleT:
        """Get the handle needed for cleanup.

        Args:
            container_name: Name of the container
            process_handle: Process handle from subprocess.Popen

        Returns:
            CleanupHandle instance for this runtime
        """
        pass


class DockerRuntime(ContainerRuntime[DockerCleanupHandle]):
    """Docker container runtime implementation."""

    def build_run_command(
        self,
        image: str,
        container_name: str,
        gpu_ids: list[int],
        env_vars: dict[str, str],
        bind_mounts: list[tuple[str, str, str]],
        command: list[str],
    ) -> list[str]:
        cmd = ["docker", "run"]

        # GPU access
        gpu_str = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        cmd.extend(["--gpus", f'"device={gpu_str}"'])

        # Namespace sharing
        cmd.extend(["--ipc", "host"])
        cmd.extend(["--pid", "host"])
        cmd.extend(["--net", "host"])

        # Container name
        cmd.extend(["--name", container_name])

        # Environment variables
        for key, value in env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Bind mounts
        for host_path, container_path, mode in bind_mounts:
            if mode:
                cmd.extend(["-v", f"{host_path}:{container_path}:{mode}"])
            else:
                cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Image
        cmd.append(image)

        # Command
        cmd.extend(command)

        return cmd

    def get_cleanup_handle(
        self, container_name: str, process_handle: subprocess.Popen | None
    ) -> DockerCleanupHandle:
        return DockerCleanupHandle(container_name)


class SingularityRuntime(ContainerRuntime[SingularityCleanupHandle]):
    """Singularity container runtime implementation."""

    def build_run_command(
        self,
        image: str,
        container_name: str,
        gpu_ids: list[int],
        env_vars: dict[str, str],
        bind_mounts: list[tuple[str, str, str]],
        command: list[str],
    ) -> list[str]:
        cmd = ["singularity", "exec"]

        # GPU access (Singularity uses --nv for NVIDIA GPUs)
        cmd.append("--nv")

        # Environment variables
        # Always include PYTHONNOUSERSITE=1 for Singularity
        all_env_vars = {"PYTHONNOUSERSITE": "1", **env_vars}
        for key, value in all_env_vars.items():
            cmd.extend(["--env", f"{key}={value}"])

        # Bind mounts
        for host_path, container_path, mode in bind_mounts:
            if mode:
                cmd.extend(["--bind", f"{host_path}:{container_path}:{mode}"])
            else:
                cmd.extend(["--bind", f"{host_path}:{container_path}"])

        # Image (.sif file)
        cmd.append(image)

        # Command - for vLLM, we need to use "vllm serve" instead of the docker entrypoint
        # The command passed in will have the model_id as the first argument
        if command:
            # Convert vLLM Docker entrypoint style to direct command
            cmd.extend(["vllm", "serve"])
            cmd.extend(command)

        return cmd

    def get_cleanup_handle(
        self, container_name: str, process_handle: subprocess.Popen | None
    ) -> SingularityCleanupHandle:
        if process_handle is None:
            raise ValueError("Process handle is required for Singularity cleanup")
        return SingularityCleanupHandle(process_handle)
