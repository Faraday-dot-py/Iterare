"""TIDE batch job tool — CSU TIDE cluster via JupyterHub + Jupyter Server API."""

from .client import TIDEClient
from .jobs import run_script, run_code, gpu_info, JobResult

__all__ = ["TIDEClient", "run_script", "run_code", "gpu_info", "JobResult"]
