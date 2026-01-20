"""
Workspace Management for Council v2 - Phase 5.

WHAT THIS FILE DOES:
-------------------
Manages the output directory structure for council sessions.
Creates organized project layouts, tracks files, and provides
utilities for the execution engine.

KEY DIFFERENCES FROM V1:
-----------------------
- Simplified: No per-agent directories (v2 uses tool-based execution)
- Integrates with execution.py's ToolImplementations
- Respects --output flag from CLI
- Session-based organization with timestamps

DIRECTORY STRUCTURE:
-------------------
<output_base>/
├── session_<id>_<timestamp>/
│   ├── src/           # Source code
│   ├── tests/         # Test files
│   ├── docs/          # Documentation
│   ├── config/        # Configuration files
│   └── workspace.json # Session metadata
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class WorkspaceManager:
    """
    Manages workspace directories for council sessions.

    Provides:
    - Automatic directory structure creation
    - File organization by type
    - Session metadata tracking
    - Tree view generation for UI

    Usage:
        workspace = WorkspaceManager(Path("./output"), "abc123")
        workspace.setup()
        workspace.write_file("src/main.py", "print('hello')")
        print(workspace.get_structure())
        final_path = workspace.finalize()
    """

    # Standard project directories
    STANDARD_DIRS = ["src", "tests", "docs", "config"]

    # File type to directory mapping
    FILE_TYPE_DIRS = {
        ".py": "src",
        ".js": "src",
        ".ts": "src",
        ".go": "src",
        ".rs": "src",
        ".java": "src",
        ".rb": "src",
        ".test.py": "tests",
        ".test.js": "tests",
        ".test.ts": "tests",
        "_test.py": "tests",
        "_test.go": "tests",
        ".spec.js": "tests",
        ".spec.ts": "tests",
        ".md": "docs",
        ".rst": "docs",
        ".txt": "docs",
        ".yaml": "config",
        ".yml": "config",
        ".json": "config",
        ".toml": "config",
        ".ini": "config",
        ".env": "config",
    }

    def __init__(
        self,
        output_path: Path,
        session_id: str,
        include_timestamp: bool = True
    ):
        """
        Initialize workspace manager.

        Args:
            output_path: Base directory for output
            session_id: Unique session identifier
            include_timestamp: Whether to include timestamp in directory name
        """
        self.output_base = Path(output_path).resolve()
        self.session_id = session_id
        self.include_timestamp = include_timestamp

        # Create session directory name
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir_name = f"session_{session_id}_{timestamp}"
        else:
            self.session_dir_name = f"session_{session_id}"

        self.workspace_path = self.output_base / self.session_dir_name

        # Tracking
        self._is_setup = False
        self._created_files: list[str] = []
        self._modified_files: list[str] = []
        self._metadata: dict = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "files": [],
            "directories": [],
        }

    @property
    def path(self) -> Path:
        """Get the workspace path."""
        return self.workspace_path

    def setup(self) -> Path:
        """
        Create the workspace directory structure.

        Creates:
        - Base workspace directory
        - Standard subdirectories (src/, tests/, docs/, config/)
        - Metadata file

        Returns:
            Path to the workspace root
        """
        if self._is_setup:
            return self.workspace_path

        # Create base directory
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Create standard directories
        for dir_name in self.STANDARD_DIRS:
            dir_path = self.workspace_path / dir_name
            dir_path.mkdir(exist_ok=True)
            self._metadata["directories"].append(dir_name)

        # Write initial metadata
        self._write_metadata()

        self._is_setup = True
        return self.workspace_path

    def _write_metadata(self) -> None:
        """Write workspace metadata to file."""
        metadata_path = self.workspace_path / "workspace.json"
        self._metadata["updated_at"] = datetime.now().isoformat()
        self._metadata["files"] = self._created_files.copy()

        with open(metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def _get_target_dir(self, path: str) -> str:
        """
        Determine target directory for a file based on its extension.

        Args:
            path: Relative file path

        Returns:
            Target directory name or empty string for root
        """
        path_lower = path.lower()

        # Check compound extensions first (e.g., .test.py)
        for ext, dir_name in sorted(
            self.FILE_TYPE_DIRS.items(),
            key=lambda x: -len(x[0])  # Longer extensions first
        ):
            if path_lower.endswith(ext):
                return dir_name

        return ""  # Root directory

    def _resolve_path(self, path: str) -> Path:
        """
        Resolve a path within the workspace, preventing directory traversal.

        Args:
            path: Relative path within workspace

        Returns:
            Absolute path guaranteed to be within workspace

        Raises:
            ValueError: If path attempts to escape workspace
        """
        # Remove leading slashes to treat as relative
        path = path.lstrip("/")

        # Resolve the full path
        full_path = (self.workspace_path / path).resolve()

        # Ensure it's still within workspace
        try:
            full_path.relative_to(self.workspace_path.resolve())
        except ValueError:
            raise ValueError(f"Path '{path}' attempts to escape workspace")

        return full_path

    def write_file(
        self,
        path: str,
        content: str,
        auto_organize: bool = True
    ) -> str:
        """
        Write content to a file in the workspace.

        Args:
            path: Relative path (may be reorganized)
            content: File content
            auto_organize: If True, auto-place in appropriate directory

        Returns:
            Final relative path where file was written
        """
        if not self._is_setup:
            self.setup()

        # Optionally reorganize into standard directories
        if auto_organize:
            # If path doesn't already have a directory prefix
            if "/" not in path and "\\" not in path:
                target_dir = self._get_target_dir(path)
                if target_dir:
                    path = f"{target_dir}/{path}"

        # Resolve and validate path
        full_path = self._resolve_path(path)

        # Create parent directories
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if modifying existing file
        is_new = not full_path.exists()

        # Write content
        full_path.write_text(content)

        # Track
        rel_path = str(full_path.relative_to(self.workspace_path))
        if is_new:
            if rel_path not in self._created_files:
                self._created_files.append(rel_path)
        else:
            if rel_path not in self._modified_files:
                self._modified_files.append(rel_path)

        # Update metadata
        self._write_metadata()

        return rel_path

    def read_file(self, path: str) -> str:
        """
        Read a file from the workspace.

        Args:
            path: Relative path within workspace

        Returns:
            File contents

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        full_path = self._resolve_path(path)

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return full_path.read_text()

    def file_exists(self, path: str) -> bool:
        """Check if a file exists in the workspace."""
        try:
            full_path = self._resolve_path(path)
            return full_path.exists()
        except ValueError:
            return False

    def list_files(self, pattern: str = "*") -> list[str]:
        """
        List files in the workspace matching a pattern.

        Args:
            pattern: Glob pattern (default: all files)

        Returns:
            List of relative file paths
        """
        if not self.workspace_path.exists():
            return []

        files = []
        for file_path in self.workspace_path.rglob(pattern):
            if file_path.is_file() and file_path.name != "workspace.json":
                rel_path = str(file_path.relative_to(self.workspace_path))
                files.append(rel_path)

        return sorted(files)

    def get_structure(self, max_depth: int = 3) -> str:
        """
        Get a tree view of the workspace structure.

        Args:
            max_depth: Maximum depth to traverse

        Returns:
            ASCII tree representation
        """
        if not self.workspace_path.exists():
            return "(workspace not yet created)"

        lines = [self.session_dir_name + "/"]
        self._build_tree(self.workspace_path, "", lines, 0, max_depth)
        return "\n".join(lines)

    def _build_tree(
        self,
        path: Path,
        prefix: str,
        lines: list[str],
        depth: int,
        max_depth: int
    ) -> None:
        """Build tree representation recursively."""
        if depth >= max_depth:
            return

        # Get children, excluding hidden files and metadata
        children = sorted([
            p for p in path.iterdir()
            if not p.name.startswith(".") and p.name != "workspace.json"
        ], key=lambda p: (not p.is_dir(), p.name.lower()))

        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "

            if child.is_dir():
                lines.append(f"{prefix}{connector}{child.name}/")
                new_prefix = prefix + ("    " if is_last else "│   ")
                self._build_tree(child, new_prefix, lines, depth + 1, max_depth)
            else:
                lines.append(f"{prefix}{connector}{child.name}")

    def finalize(self) -> Path:
        """
        Finalize the workspace and return the output path.

        Updates metadata with final statistics.

        Returns:
            Path to the workspace root
        """
        self._metadata["finalized_at"] = datetime.now().isoformat()
        self._metadata["total_files"] = len(self._created_files)
        self._metadata["modified_files"] = len(self._modified_files)
        self._write_metadata()

        return self.workspace_path

    def get_stats(self) -> dict:
        """
        Get workspace statistics.

        Returns:
            Dict with file counts and sizes
        """
        total_size = 0
        file_count = 0

        if self.workspace_path.exists():
            for file_path in self.workspace_path.rglob("*"):
                if file_path.is_file():
                    file_count += 1
                    total_size += file_path.stat().st_size

        return {
            "path": str(self.workspace_path),
            "session_id": self.session_id,
            "file_count": file_count,
            "total_size_bytes": total_size,
            "created_files": len(self._created_files),
            "modified_files": len(self._modified_files),
        }


def create_workspace(
    output_path: Optional[Path] = None,
    session_id: Optional[str] = None
) -> WorkspaceManager:
    """
    Create a new workspace with defaults.

    Args:
        output_path: Base output directory (default: ./council-output)
        session_id: Session ID (generated if not provided)

    Returns:
        Configured WorkspaceManager
    """
    import uuid

    output_path = output_path or Path("./council-output")
    session_id = session_id or str(uuid.uuid4())[:8]

    workspace = WorkspaceManager(output_path, session_id)
    workspace.setup()

    return workspace


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import tempfile

    # Test workspace creation
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = WorkspaceManager(Path(tmpdir), "test123")
        workspace.setup()

        # Write some files
        workspace.write_file("main.py", "print('hello')")
        workspace.write_file("test_main.py", "def test_main(): pass")
        workspace.write_file("README.md", "# Project")
        workspace.write_file("config.yaml", "key: value")
        workspace.write_file("src/utils.py", "# utilities")

        # Show structure
        print("Workspace structure:")
        print(workspace.get_structure())

        # Show stats
        print("\nStats:")
        for key, value in workspace.get_stats().items():
            print(f"  {key}: {value}")

        # List files
        print("\nFiles:")
        for f in workspace.list_files():
            print(f"  {f}")

        workspace.finalize()
        print("\nWorkspace finalized!")
