Repository structure
====================

* ``oos_review/`` – source code (pipeline libraries, scripts, review API, dashboard)
* ``docs/`` – Sphinx documentation scaffold
* ``data/`` – challenge inputs, pipeline outputs, and baseline snapshot

Python entry points live under ``oos_review/scripts/``.
The Hencheck UI is ``oos_review/frontend/`` and the FastAPI service is
``oos_review/backend/``.
