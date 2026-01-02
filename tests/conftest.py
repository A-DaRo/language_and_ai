"""Pytest fixtures shared across the suite.

This project runs heavyweight transformer models in integration tests.
On low-VRAM laptops (e.g., 4GB), memory fragmentation/leakage between tests
can cause spurious CUDA OOMs.
"""

import gc

import pytest


@pytest.fixture(autouse=True)
def _aggressive_resource_cleanup():
	"""Force Python + CUDA cleanup after every test function."""
	yield

	gc.collect()

	try:
		import torch

		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			# Helps release inter-process cached allocations when supported.
			try:
				torch.cuda.ipc_collect()
			except Exception:
				pass
	except Exception:
		# Tests may run without torch (or without CUDA); keep fixture safe.
		pass
