"""Abstract container runtime interface for Docker and Singularity."""

from __future__ import annotations

import subprocess
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CleanupHandle(ABC):
    """Abstract base class for cleanup handles."""

    @abstractmethod
    def cleanup(self) -> None:
        """Start cleanup of the container/process (non-blocking)."""
        pass

    @abstractmethod
    def wait(self) -> None:
        """Wait for cleanup to complete."""
        pass


class DockerCleanupHandle(CleanupHandle):
    """Cleanup handle for Docker containers."""

    def __init__(self, container_name: str, docker_cmd: list[str]) -> None:
        """Initialize the cleanup handle."""
        self.container_name = container_name
        self._cleanup_process: subprocess.Popen | None = None
        self._docker_cmd = docker_cmd

    def cleanup(self) -> None:
        """Start cleanup of the container/process (non-blocking)."""
        logger.info(f"Removing Docker container: {self.container_name}")
        self._cleanup_process = subprocess.Popen(
            [*self._docker_cmd, "rm", "-f", self.container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def wait(self) -> None:
        """Wait for cleanup to complete."""
        if self._cleanup_process is not None:
            self._cleanup_process.wait()
            logger.info(f"Docker container {self.container_name} removed.")
        else:
            logger.warning("Cleanup was not started yet.")


class SingularityCleanupHandle(CleanupHandle):
    """Cleanup handle for Singularity processes."""

    def __init__(self, process_handle: subprocess.Popen) -> None:
        """Initialize the cleanup handle."""
        self.process_handle = process_handle
        self._cleanup_started = False

    def cleanup(self) -> None:
        """Start cleanup of the container/process (non-blocking)."""
        logger.info("Terminating Singularity exec process.")
        self.process_handle.terminate()
        self._cleanup_started = True

    def wait(self) -> None:
        """Wait for cleanup to complete."""
        if not self._cleanup_started:
            logger.warning("Cleanup was not started yet.")
            return

        try:
            self.process_handle.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.info("Singularity process did not terminate gracefully, killing it.")
            self.process_handle.kill()
            self.process_handle.wait()
        logger.info("Singularity exec process terminated.")


class ContainerRuntime(ABC):
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
            command: Command to run inside the container. This should be the *full command*
                including whatever entrypoint that the image may already have.

        Returns:
            Command as list of strings
        """
        pass

    @abstractmethod
    def get_cleanup_handle(
        self, container_name: str, process_handle: subprocess.Popen | None
    ) -> CleanupHandle:
        """Get the handle needed for cleanup.

        Args:
            container_name: Name of the container
            process_handle: Process handle from subprocess.Popen

        Returns:
            CleanupHandle instance for this runtime
        """
        pass


class DockerRuntime(ContainerRuntime):
    """Docker container runtime implementation."""

    def __init__(self) -> None:
        """Initialize the Docker runtime."""
        try:
            subprocess.check_call(args=["docker", "ps"])
            self._docker_cmd = ["docker"]
        except subprocess.CalledProcessError:
            self._docker_cmd = ["sudo", "docker"]

    def build_run_command(
        self,
        image: str,
        container_name: str,
        gpu_ids: list[int],
        env_vars: dict[str, str],
        bind_mounts: list[tuple[str, str, str]],
        command: list[str],
    ) -> list[str]:
        cmd = [*self._docker_cmd, "run"]

        # GPU access
        gpu_str = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        cmd.extend(["--gpus", f'"device={gpu_str}"'])

        # Namespace sharing
        cmd.extend(["--ipc", "host"])
        cmd.extend(["--net", "host"])
        # cmd.extend(["--pid", "host"])  # XXX(J1): Required for CUDA IPC?

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

        # Wipe out entrypiont. This is done because Singularity does not support entrypoints,
        # and thus we asked the user to provide the full command to run inside the container.
        cmd.extend(["--entrypoint", ""])

        # Image
        cmd.append(image)

        # Command
        cmd.extend(command)

        return cmd

    def get_cleanup_handle(
        self, container_name: str, process_handle: subprocess.Popen | None
    ) -> DockerCleanupHandle:
        return DockerCleanupHandle(container_name, self._docker_cmd)


class SingularityRuntime(ContainerRuntime):
    """Singularity container runtime implementation.

    Singularity
    - Does not support entrypoints; the full command must be specified.
    - GPU support is enabled with the `--nv` flag.
    - Mounts the current user's home directory by default, and the container process
        runs in the foreground under the current user's account.

    Therefore, compared to Docker
    - We ask the user to provide the full command to run inside the container,
        and we wipe out the entrypoint in the Docker tuneime.
    - We always add the `--nv` flag for GPU support.
    - We avoid any volume mounts that have a hardcoded user home directory path,
        like `/root/.cache`.
    """

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
        # The user's home directory is mounted by default, and user site-packages
        # may interfere with the container environment, e.g., link errors of PyTorch
        # built against different CUDA versions. Setting PYTHONNOUSERSITE=1 fixed that.
        all_env_vars = {"PYTHONNOUSERSITE": "1", **env_vars}

        # All GPUs are visible inside the container by default, so we need to set this
        # explicitly to limit visible GPUs.
        if gpu_ids:
            gpu_ids_str = ",".join(str(gpu_id) for gpu_id in gpu_ids)
            all_env_vars["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

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

        if command:
            cmd.extend(command)

        return cmd

    def get_cleanup_handle(
        self, container_name: str, process_handle: subprocess.Popen | None
    ) -> SingularityCleanupHandle:
        if process_handle is None:
            raise ValueError("Process handle is required for Singularity cleanup")
        return SingularityCleanupHandle(process_handle)
