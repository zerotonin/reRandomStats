Installation
============

From Source (Recommended)
-------------------------

.. code-block:: bash

   git clone https://github.com/zerotonin/rerandomstats.git
   cd rerandomstats
   pip install -e ".[dev]"

Via Conda
---------

.. code-block:: bash

   conda env create -f environment.yml
   conda activate rerandomstats
   pip install -e .

Dependencies
------------

Core dependencies are installed automatically:

- ``numpy >= 1.22``
- ``scipy >= 1.9``
- ``pandas >= 1.5``
- ``statsmodels >= 0.13``
- ``prettytable >= 3.0``
- ``tqdm >= 4.60``

Development extras (``pip install -e ".[dev]"``):

- ``pytest``, ``pytest-cov``
- ``sphinx``, ``sphinx-rtd-theme``
- ``ruff``
