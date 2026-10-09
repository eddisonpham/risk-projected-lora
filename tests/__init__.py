"""Test package marker (deliberately present, not decorative).

``tests`` must be a **regular** package: ``draccus`` (pulled in by lerobot) ships an
empty top-level ``tests`` package in site-packages, and per PEP 420 a regular package
found anywhere on ``sys.path`` beats a namespace-package portion found earlier — which
would silently shadow this directory and break ``from tests.tolerances import ...``.
``tests/conftest.py`` also puts the repository root first on ``sys.path`` when the
suite runs through ``pytest``.
"""
