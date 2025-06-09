:layout: landing
:description: TGP is a library for pooling in Graph Neural Networks.
:dark_code: true

Torch Geometric Pool
====================

.. rst-class:: lead

    The graph pooling library made for :pyg:`null` `PyTorch Geometric
    <https://github.com/pyg-team/pytorch_geometric/>`_.

.. container:: buttons

    `Docs <content/quickstart.html>`_
    `GitHub <https://github.com/tgp-team/torch-geometric-pool>`_

:tgp:`null` **tgp (Torch Geometric Pool)** is a library built on top of :pyg:`null` `PyTorch Geometric
<https://github.com/pyg-team/pytorch_geometric/>`_ that brings every major **graph-pooling operator**
into a single, unified framework. Drop-in layers let you turn any
vanilla GNN into a **hierarchical** one with just a few lines of code.


.. admonition:: Want to learn more about graph pooling?
   :class: tip

   Read our series of `blog posts <https://gnn-pooling.notion.site/>`_ on the topic.

Features
--------

.. grid:: 1 1 1 1
    :padding: 0
    :gutter: 2

    .. grid-item-card:: :octicon:`file-code;1em;sd-text-primary` Unified API
        :link: content/src
        :link-type: doc
        :shadow: sm

        All pooling layers in :tgp:`tgp` are implemented following the
        :class:`~tgp.src.SRCPooling` (:class:`~tgp.select.Select` →
        :class:`~tgp.reduce.Reduce` → :class:`~tgp.connect.Connect` → :class:`~tgp.lift.Lift`)
        framework, introduced in `Understanding Pooling in Graph Neural Networks
        <https://arxiv.org/abs/2110.05292>`_, which ensures a consistent API across all
        methods and seamless interoperability.

    .. grid-item-card:: :octicon:`archive;1em;sd-text-primary` All your pooling operators in one place.
        :link: api/poolers
        :link-type: doc
        :shadow: sm

        Choose from a variety of pooling methods, including sparse techniques like
        :class:`~tgp.poolers.TopkPooling`, :class:`~tgp.poolers.NDPPooling`,
        :class:`~tgp.poolers.GraclusPooling`, and dense methods such as
        :class:`~tgp.poolers.DiffPool` and :class:`~tgp.poolers.MinCutPooling`.

    .. grid-item-card:: :octicon:`repo-forked;1em;sd-text-primary` Tweak and create new pooling layers.
        :link: api/src
        :link-type: doc
        :shadow: sm

        Thanks to the modular :class:`~tgp.src.SRCPooling` framework, the components of
        different pooling layers :tgp:`tgp` can be easily combined with each other,
        replaced with existing modules or with completely new ones.

    .. grid-item-card:: :octicon:`zap;1em;sd-text-primary` Precomputed & On-the-Fly Pooling.
        :link: api/data/index
        :link-type: doc
        :shadow: sm

        Accelerate training by precomputing the coarse graph (assignments and connectivity)
        for methods like :class:`~tgp.poolers.NDPPooling` and
        :class:`~tgp.poolers.GraclusPooling`. Alternatively, use on-the-fly pooling
        (e.g., :class:`~tgp.poolers.TopkPooling` or :class:`~tgp.poolers.MinCutPooling`) that
        computes assignments dynamically and supports end-to-end gradient flow.


.. rst-class:: lead

    Not familiar with Select-Reduce-Connect? Read our introductory notes about
    :octicon:`book` :doc:`content/src`.

Installation
------------

:tgp:`tgp` is compatible with Python>=3.9. We recommend installation
on a `Anaconda or Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install>`_
environment or a `virtual env <https://docs.python.org/3/library/venv.html>`_.
:tgp:`tgp` is conveniently available as a Python package on PyPI and
can be easily installed using pip.

.. code-block:: bash

    pip install torch-geometric-pool

For the latest version, consider installing from source:

.. code-block:: bash

    pip install git+https://github.com/tgp-team/torch-geometric-pool.git


.. admonition:: Before installation
   :class: caution

   :tgp:`tgp` requires `PyTorch>=1.8 <https://pytorch.org/>`_ and
   `PyG>=2.0 <https://github.com/pyg-team/pytorch_geometric/>`_ to be present in your
   environment.


Quick example
-------------

.. code-block:: python

    import torch
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    from tgp.poolers import get_pooler

    # Create a simple graph (5 nodes, 5 edges)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0],
                               [1, 0, 3, 2, 0, 4]], dtype=torch.long)
    x = torch.randn((5, 16))  # node features

    data = Data(x=x, edge_index=edge_index)

    # Instantiate a Top-K pooling layer via its alias
    pool = get_pooler("topk", in_channels=32, ratio=0.5)

    # Forward pass with pooling
    out = GCNConv(in_channels=16, out_channels=32)(data.x, data.edge_index)
    out = pool(out, data.edge_index, batch=None)  # PoolingOutput(so=[5, 3], x=[3, 32], edge_index=[2, 2])
    out = GCNConv(in_channels=32, out_channels=32)(out.x, out.edge_index)

For a deeper dive, head over to the :doc:`content/quickstart` section, browse
the :doc:`tutorials/index`, and explore our :doc:`APIs <api/tgp>`.


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   content/quickstart
   content/src
   tutorials/index


.. toctree::
   :glob:
   :maxdepth: 3
   :caption: API
   :hidden:

   api/tgp
   api/src
   api/select
   api/reduce
   api/lift
   api/connect
   api/poolers
   api/datasets
   api/data/index
   api/utils/index


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Indices
   :hidden:

   genindex
