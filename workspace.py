"""
Workspace management for Agent Council.
Handles directory creation, file operations, and project organization.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


class WorkspaceManager:
    """Manages workspace directories for agent work."""
    
    def __init__(self, base_path: str, session_id: str):
        self.base_path = Path(base_path).expanduser()
        self.session_id = session_id
        self.session_path = self.base_path / session_id
        self.agent_dirs: dict[str, Path] = {}
        
    def setup(self):
        """Create the main workspace directory."""
        self.session_path.mkdir(parents=True, exist_ok=True)
        
        # Create standard subdirectories
        (self.session_path / "src").mkdir(exist_ok=True)
        (self.session_path / "tests").mkdir(exist_ok=True)
        (self.session_path / "docs").mkdir(exist_ok=True)
        
        # Create metadata file
        metadata = {
            "session_id": self.session_id,
            "created": datetime.now().isoformat(),
            "status": "in_progress"
        }
        self._write_json(self.session_path / "council_metadata.json", metadata)
        
        return self.session_path
        
    def setup_agent_directories(self, agent_names: list[str]):
        """Create separate directories for each agent."""
        self.setup()
        
        agents_dir = self.session_path / "_agents"
        agents_dir.mkdir(exist_ok=True)
        
        for name in agent_names:
            safe_name = name.lower().replace(" ", "_")
            agent_dir = agents_dir / safe_name
            agent_dir.mkdir(exist_ok=True)
            self.agent_dirs[name] = agent_dir
            
        return self.agent_dirs
        
    def get_agent_dir(self, agent_name: str) -> Path:
        """Get the directory for a specific agent."""
        if agent_name in self.agent_dirs:
            return self.agent_dirs[agent_name]
        
        # Create on demand if not exists
        safe_name = agent_name.lower().replace(" ", "_")
        agent_dir = self.session_path / "_agents" / safe_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        self.agent_dirs[agent_name] = agent_dir
        return agent_dir
        
    def write_agent_file(self, agent_name: str, filename: str, content: str):
        """Write a file to an agent's directory."""
        agent_dir = self.get_agent_dir(agent_name)
        filepath = agent_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)
        return filepath
        
    def read_agent_file(self, agent_name: str, filename: str) -> Optional[str]:
        """Read a file from an agent's directory."""
        agent_dir = self.get_agent_dir(agent_name)
        filepath = agent_dir / filename
        if filepath.exists():
            return filepath.read_text()
        return None
        
    def list_agent_files(self, agent_name: str) -> list[Path]:
        """List all files in an agent's directory."""
        agent_dir = self.get_agent_dir(agent_name)
        return list(agent_dir.rglob("*"))
        
    def merge_agent_outputs(self, output_dir: Optional[Path] = None) -> Path:
        """Merge all agent outputs into the main project structure."""
        target = output_dir or self.session_path / "merged"
        target.mkdir(parents=True, exist_ok=True)
        
        for agent_name, agent_dir in self.agent_dirs.items():
            for file in agent_dir.rglob("*"):
                if file.is_file():
                    # Determine target location based on file type
                    rel_path = file.relative_to(agent_dir)
                    
                    if file.suffix in [".py", ".js", ".ts", ".jsx", ".tsx"]:
                        dest = target / "src" / rel_path
                    elif file.suffix in [".test.py", ".test.js", ".spec.ts"]:
                        dest = target / "tests" / rel_path
                    elif file.suffix in [".md", ".txt", ".rst"]:
                        dest = target / "docs" / rel_path
                    else:
                        dest = target / rel_path
                        
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, dest)
                    
        return target
        
    def finalize(self) -> Path:
        """Finalize the workspace and return the output path."""
        # Merge agent outputs
        merged = self.merge_agent_outputs()
        
        # Update metadata
        metadata_file = self.session_path / "council_metadata.json"
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
        else:
            metadata = {}
            
        metadata["status"] = "completed"
        metadata["completed"] = datetime.now().isoformat()
        metadata["output_path"] = str(merged)
        
        self._write_json(metadata_file, metadata)
        
        return merged
        
    def cleanup(self):
        """Remove the workspace directory."""
        if self.session_path.exists():
            shutil.rmtree(self.session_path)
            
    def _write_json(self, filepath: Path, data: dict):
        """Write JSON to a file."""
        filepath.write_text(json.dumps(data, indent=2))
        
    def get_structure(self) -> str:
        """Get a string representation of the workspace structure."""
        lines = [f"Workspace: {self.session_path}"]
        
        def add_tree(path: Path, prefix: str = ""):
            items = sorted(path.iterdir())
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{item.name}")
                
                if item.is_dir():
                    extension = "    " if is_last else "│   "
                    add_tree(item, prefix + extension)
                    
        if self.session_path.exists():
            add_tree(self.session_path)
            
        return "\n".join(lines)
