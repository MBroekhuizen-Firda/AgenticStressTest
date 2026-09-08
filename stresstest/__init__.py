"""Agentic coding stress test harness.

Measures how many concurrent students a single GPU running a local LLM can
serve before the experience becomes unacceptable, and how much VRAM that
actually takes.

Code and identifiers are English on purpose; the documentation and the
result report are Dutch because they are read by colleagues and management.
"""

__version__ = "1.0.0"
