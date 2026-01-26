"""
Alert system for Agent Council.
Supports terminal notifications and macOS system notifications.
"""

import subprocess
import sys
from datetime import datetime


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, terminal: bool = True, macos: bool = True):
        self.terminal_enabled = terminal
        self.macos_enabled = macos
        
    def _terminal_bell(self):
        """Ring terminal bell."""
        print("\a", end="", flush=True)
        
    def _terminal_alert(self, title: str, message: str, style: str = "warning"):
        """Display a prominent terminal alert."""
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console()
        
        style_map = {
            "critical": "bold red",
            "warning": "bold yellow",
            "info": "bold blue",
            "success": "bold green"
        }
        
        border_style = style_map.get(style, "bold yellow")
        
        console.print()
        console.print(Panel(
            f"[{border_style}]{message}[/{border_style}]",
            title=f"⚠️  {title}",
            border_style=border_style
        ))
        console.print()
        
        # Ring bell for critical
        if style == "critical":
            self._terminal_bell()
            
    def _macos_notification(self, title: str, message: str, sound: bool = True):
        """Send macOS notification using osascript."""
        # Escape quotes in message
        title = title.replace('"', '\\"')
        message = message.replace('"', '\\"')
        
        script = f'''
        display notification "{message}" with title "Agent Council" subtitle "{title}"
        '''
        
        if sound:
            script += '\nbeep'
            
        try:
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5
            )
        except Exception as e:
            print(f"[dim]Could not send macOS notification: {e}[/dim]")
            
    def info(self, title: str, message: str = ""):
        """Send an info-level alert."""
        if self.terminal_enabled:
            self._terminal_alert(title, message or title, "info")
            
    def warning(self, title: str, message: str = ""):
        """Send a warning-level alert."""
        if self.terminal_enabled:
            self._terminal_alert(title, message or title, "warning")
            
    def critical(self, title: str, message: str = ""):
        """Send a critical alert (terminal + macOS)."""
        full_message = message or title
        
        if self.terminal_enabled:
            self._terminal_alert(title, full_message, "critical")
            
        if self.macos_enabled:
            self._macos_notification(title, full_message, sound=True)
            
    def success(self, title: str, message: str = ""):
        """Send a success notification."""
        if self.terminal_enabled:
            self._terminal_alert(title, message or title, "success")
            
    def progress(self, message: str):
        """Show a progress message (no alert, just info)."""
        from rich.console import Console
        Console().print(f"[dim]→ {message}[/dim]")


# Standalone test
if __name__ == "__main__":
    alerts = AlertManager(terminal=True, macos=True)
    
    print("Testing alerts...\n")
    
    alerts.info("Info Alert", "This is an informational message")
    alerts.warning("Warning Alert", "Something might need attention")
    alerts.success("Success Alert", "Operation completed successfully")
    alerts.critical("Critical Alert", "Immediate attention required!")
    
    print("\nAlert test complete.")
