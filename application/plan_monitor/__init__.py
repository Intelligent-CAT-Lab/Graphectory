"""Plan Monitor - Independent phase tracking for agent actions."""

__version__ = "0.1.0"

# Export main interfaces for easy integration
from plan_monitor.monitor import StatefulPhaseMonitor
from plan_monitor.phases import ActionEvent, MonitorResult

__all__ = ["StatefulPhaseMonitor", "ActionEvent", "MonitorResult", "__version__"]
