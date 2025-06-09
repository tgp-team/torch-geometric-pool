:description: Here is the guide on how to install Torch Geometric Pool.

Installation
============

.. rst-class:: lead

   Install :tgp:`Torch Geometric Pool` in your graph deep learning pipeline.

----

Torch Geometric Pool is compatible with Python>=3.9. We recommend installation
on a `Anaconda or Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install>`_
environment or a `virtual env <https://docs.python.org/3/library/venv.html>`_.

Torch Geometric Pool is conveniently available as a Python package on PyPI and 
can be easily installed using pip.

.. code-block:: shell

    pip install torch-geometric-pool

For the latest version, consider installing from source:

.. code-block:: shell

    pip install git+https://github.com/tgp-team/torch-geometric-pool.git

.. admonition:: Before installation
   :class: caution

   :tgp:`Torch Geometric Pool` is built upon `PyTorch>=1.8 <https://pytorch.org/>`_ and
   `PyG>=2.0 <https://github.com/pyg-team/pytorch_geometric/>`_. Make sure you have
   both installed in your environment before installation. In the following,
   we provide instructions on how to install them for the chosen installation
   procedure.


Installing using conda
----------------------

To install Torch Geometric Pool using conda, clone the repository, navigate to the library root
directory and create a new conda environment using the provided conda configuration:

.. code:: shell

    git clone https://github.com/tgp-team/torch-geometric-pool.git
    cd torch-geometric-pool
    conda env create -f conda_env.yml

Then, activate the environment and install :code:`tgp` using :code:`pip`.

.. code:: shell

    conda activate tgp
    pip install .


Quickstart
----------

.. grid:: 1 1 2 2
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`repo;1em;sd-text-primary` Package API
        :link: ../api/tgp
        :link-type: doc
        :shadow: sm

        Learn how to configure and build your graph pooling operator.

    .. grid-item-card::  :octicon:`file-code;1em;sd-text-primary` Tutorials
        :link: ../tutorials/index
        :link-type: doc
        :shadow: sm

        Learn how to use :tgp:`tgp` at its best.