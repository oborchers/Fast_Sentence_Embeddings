"""The lifecycle core shared by every measurement: durable records, manifest, DONE, collection,
machine check, time and money budget, environments and steps of one allocation.

Isolated by rule: this package imports only the standard library and itself
(tools/guardrails_imports.py), so the bootstrap script can run its record writer on a bare
interpreter before any environment exists.
"""
