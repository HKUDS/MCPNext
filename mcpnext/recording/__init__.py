"""
    RecordingManager
      ├── internal management of platform.RecordingClient
      ├── internal management of platform.ScreenshotClient  
      ├── internal management of TrajectoryRecorder
      └── internal management of ActionRecorder
"""

# Auto-record the tool execution
from .manager import RecordingManager

# Low-level components (advanced users)
from .recorder import TrajectoryRecorder
from .action_recorder import ActionRecorder

# Utility functions
from .utils import (
    load_trajectory_from_jsonl,
    load_metadata,
    format_trajectory_for_export,
    analyze_trajectory,
    load_recording_session,
    filter_trajectory,
    extract_errors,
    generate_summary_report,
)

from .action_recorder import (
    load_agent_actions,
    analyze_agent_actions,
    format_agent_actions,
)

__all__ = [
    # Manager
    'RecordingManager',
    
    # Recorders
    'TrajectoryRecorder',
    'ActionRecorder',
    
    # Trajectory utils
    'load_trajectory_from_jsonl',
    'load_metadata',
    'format_trajectory_for_export',
    'analyze_trajectory',
    'load_recording_session',
    'filter_trajectory',
    'extract_errors',
    'generate_summary_report',
    
    # Agent action utils
    'load_agent_actions',
    'analyze_agent_actions',
    'format_agent_actions',
]