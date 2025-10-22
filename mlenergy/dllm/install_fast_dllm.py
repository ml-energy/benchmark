#!/usr/bin/env python3
"""Fix imports in Fast-dLLM files to use the fast_dllm package namespace.

This script converts old-style relative imports to proper package imports.
"""

import logging
from pathlib import Path
import re
import subprocess

logger = logging.getLogger("mlenergy.dllm.install_fast_dllm")


def fix_file_imports(file_path: Path, parent_dir: str) -> bool:
    """Fix imports in a single Python file.

    Args:
        file_path: Path to the Python file.
        parent_dir: Parent directory name ('llada' or 'dream').

    Returns:
        True if file was modified, False otherwise.
    """
    content = file_path.read_text()
    original_content = content

    # Common patterns for both llada and dream
    if parent_dir == "llada":
        # LLaDA-specific patterns
        # Pattern 1: from generate import X -> from fast_dllm.llada.generate import X
        content = re.sub(
            r"^from generate import",
            "from fast_dllm.llada.generate import",
            content,
            flags=re.MULTILINE,
        )

        # Pattern 2: from model.modeling_llada import X -> from fast_dllm.llada.model.modeling_llada import X
        content = re.sub(
            r"^from model\.modeling_llada import",
            "from fast_dllm.llada.model.modeling_llada import",
            content,
            flags=re.MULTILINE,
        )

        # Pattern 3: from model.configuration_llada import X -> from fast_dllm.llada.model.configuration_llada import X
        content = re.sub(
            r"^from model\.configuration_llada import",
            "from fast_dllm.llada.model.configuration_llada import",
            content,
            flags=re.MULTILINE,
        )

    elif parent_dir == "dream":
        # DREAM-specific patterns
        # Pattern 1: from model.modeling_dream import X -> from fast_dllm.dream.model.modeling_dream import X
        content = re.sub(
            r"^from model\.modeling_dream import",
            "from fast_dllm.dream.model.modeling_dream import",
            content,
            flags=re.MULTILINE,
        )

        # Pattern 2: from model.generation_utils import X -> from fast_dllm.dream.model.generation_utils import X
        content = re.sub(
            r"^from model\.generation_utils import",
            "from fast_dllm.dream.model.generation_utils import",
            content,
            flags=re.MULTILINE,
        )

        # Pattern 3: from model.generation_utils_block import X -> from fast_dllm.dream.model.generation_utils_block import X
        content = re.sub(
            r"^from model\.generation_utils_block import",
            "from fast_dllm.dream.model.generation_utils_block import",
            content,
            flags=re.MULTILINE,
        )

        # Pattern 4: from model.configuration_dream import X -> from fast_dllm.dream.model.configuration_dream import X
        content = re.sub(
            r"^from model\.configuration_dream import",
            "from fast_dllm.dream.model.configuration_dream import",
            content,
            flags=re.MULTILINE,
        )

        # Pattern 5: from model.tokenization_dream import X -> from fast_dllm.dream.model.tokenization_dream import X
        content = re.sub(
            r"^from model\.tokenization_dream import",
            "from fast_dllm.dream.model.tokenization_dream import",
            content,
            flags=re.MULTILINE,
        )

    if content != original_content:
        file_path.write_text(content)
        return True
    return False


def main():
    """Fix imports in all Python files in the llada and dream directories."""
    base_dir = Path(__file__).parent / "Fast-dLLM"

    modified_files = []

    # Process LLaDA files
    llada_dir = base_dir / "llada"
    if llada_dir.exists():
        print(f"Processing LLaDA files in {llada_dir}")
        for py_file in llada_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            if fix_file_imports(py_file, "llada"):
                modified_files.append(py_file)
                print(f"Fixed imports in: {py_file.relative_to(base_dir)}")

    # Process DREAM files
    dream_dir = base_dir / "dream"
    if dream_dir.exists():
        print(f"\nProcessing DREAM files in {dream_dir}")
        for py_file in dream_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            if fix_file_imports(py_file, "dream"):
                modified_files.append(py_file)
                print(f"Fixed imports in: {py_file.relative_to(base_dir)}")

    if modified_files:
        print(f"\n✓ Modified {len(modified_files)} files")
    else:
        print("\n✓ No files needed modification")


def install_runtime(self) -> None:
    """Install Fast-dLLM dependencies and setup Python path."""
    repo_url = "https://github.com/ml-energy/fast-dllm"
    commit = "037942"
    base_dir = Path(__file__).resolve().parent
    fast_dir = base_dir / "Fast-dLLM"

    try:
        if not fast_dir.exists():
            logger.info("Cloning Fast-dLLM into %s", fast_dir)
            subprocess.run(["git", "clone", repo_url, str(fast_dir)], check=True)
            logger.info("Fetching and checking out commit %s in %s", commit, fast_dir)
            subprocess.run(["git", "fetch", "--all"], cwd=str(fast_dir), check=True)
            subprocess.run(["git", "checkout", commit], cwd=str(fast_dir), check=True)

            logger.info("Installing Fast-dLLM as editable package")
            subprocess.run(["uv", "pip", "install", "-e", str(fast_dir)], check=True)

    except subprocess.CalledProcessError as e:
        logger.error("Failed to install Fast-dLLM: %s", e)
        raise

    logger.info("Fast-dLLM installation complete.")


if __name__ == "__main__":
    main()
