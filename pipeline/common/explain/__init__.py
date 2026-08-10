"""Per-prediction explainability.

Answers "why does the model think this?" at two altitudes:

* ``contributions`` / ``units`` / ``families`` turn LightGBM's exact TreeSHAP
  output into signed, human-readable drivers grouped into feature families.
* ``trace`` reconstructs the arithmetic chain that produced the published
  number (experts, simplex pool, temperature, guard, market blends) without
  re-simulating anything.

Nothing in this package is on the send path. Every caller wraps it so that a
failure degrades to today's output rather than breaking a send.
"""
