"""Adapters turn one data source into the engine's common tables.

Every adapter produces the same outputs (signals, labels, inventory), so the
rest of the pipeline never needs to know where the data came from.
"""
