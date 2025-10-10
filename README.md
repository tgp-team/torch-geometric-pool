<div align="center">
    <br><br>
    <img alt="Torch Geometric Pool" src="https://raw.githubusercontent.com/tgp-team/torch-geometric-pool/refs/heads/main/docs/source/_static/img/tgp-logo-full.svg" width="65%"/>
    <h3>The library for pooling in Graph Neural Networks</h3>
    <hr>
    <p>
    <a href='https://pypi.org/project/torch-geometric-pool/'><img alt="PyPI" src="https://img.shields.io/pypi/v/torch-geometric-pool"></a>
    <img alt="PyPI - Python Version" src="https://img.shields.io/badge/python-%3E%3D3.9-blue">
    <a href='https://github.com/tgp-team/torch-geometric-pool/actions/workflows/ci.yaml'><img alt="CI status" src="https://github.com/tgp-team/torch-geometric-pool/actions/workflows/ci.yaml/badge.svg"></a>
    <a href='https://torch-geometric-pool.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/torch-geometric-pool/badge/?version=latest' alt='Documentation Status' /></a>
    <a href='https://pepy.tech/projects/torch-geometric-pool'><img src='https://static.pepy.tech/badge/torch-geometric-pool' alt='Total Downloads' /></a>
    </p>
    <p>
    📚 <a href="https://torch-geometric-pool.readthedocs.io/en/latest/">Documentation</a> - 🚀 <a href="https://torch-geometric-pool.readthedocs.io/en/latest/content/quickstart.html">Getting Started</a> - 💻 <a href="https://torch-geometric-pool.readthedocs.io/en/latest/tutorials/index.html">Introductory notebooks</a>
    </p>
</div>

<p>
<img src="https://raw.githubusercontent.com/tgp-team/torch-geometric-pool/refs/heads/main/docs/source/_static/img/tgp-logo.svg" width="20px" align="center" style="display: inline-block; height: 1.0em; width: unset; vertical-align: text-top;"/> <b>tgp</b> <em>(Torch Geometric Pool)</em> is a library that provides a broad suite of graph pooling layers to be inserted into Graph Neural Network architectures built with <a href="https://pyg.org"><img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/logos/pyg.svg" width="20px" align="center"/> PyTorch Geometric <a href="https://pyg.org"></a>.
With <img src="https://raw.githubusercontent.com/tgp-team/torch-geometric-pool/refs/heads/main/docs/source/_static/img/tgp-logo.svg" width="20px" align="center" style="display: inline-block; height: 1.0em; width: unset; vertical-align: text-top;"/> <b>tgp</b>, you can effortlessly construct hierarchical GNNs by interleaving message-passing layers with any pooling operations.
</p>

## Features

* **Unified API.**
All pooling layers in <img src="https://raw.githubusercontent.com/tgp-team/torch-geometric-pool/refs/heads/main/docs/source/_static/img/tgp-logo.svg" width="20px" align="center" style="display: inline-block; height: 1.0em; width: unset; vertical-align: text-top;"/> tgp are implemented following the SRC (Select, Reduce, Connect) framework, introduced in [Understanding Pooling in Graph Neural Networks](https://arxiv.org/abs/2110.05292), which ensures a consistent API across all methods and seamless interoperability.

* **All your pooling operators in one place.**
Choose from a variety of pooling methods, including sparse techniques like Top-K, NDPPooling, GraclusPooling, and dense methods such as Diffpool and MinCutPool. Each operator adheres to the modular SRC framework, allowing to quickly echange them within the same GNN architecture and combine with standard message-passing layers.

* **Precomputed & On-the-Fly Pooling.**
Accelerate training by precomputing the coarse graph (assignments and connectivity) for methods like NDPPooling or GraclusPooling. Alternatively, use on-the-fly pooling (e.g., Top-K or MinCut) that computes assignments dynamically and supports end-to-end gradient flow.

* **Alias-Based Instantiation.**
Quickly create any pooler by name (e.g., `"topk"`, `"ndp"`, `"diffpool"`, `"mincut"`). Pass a configuration dict for hyperparameters, and receive a fully initialized pooling layer that conforms to the unified SRC interface.

* **Tweak and create new pooling layers.**
Thanks to the modular SRC framework, the components of different pooling layers <img src="https://raw.githubusercontent.com/tgp-team/torch-geometric-pool/refs/heads/main/docs/source/_static/img/tgp-logo.svg" width="20px" align="center" style="display: inline-block; height: 1.0em; width: unset; vertical-align: text-top;"/> tgp can be easily combined with each other, replaced with existing modules or with completely new ones.

## Getting Started

If you are unfamiliar with graph pooling, we recommend checking this [introduction](https://torch-geometric-pool.readthedocs.io/en/latest/content/src.html) to the SRC framework and this [blog](https://filippomb.github.io/blogs/gnn-pool-1/) for a deeper dive into pooling in GNNs.
Before you dive into using <img src="https://raw.githubusercontent.com/tgp-team/torch-geometric-pool/refs/heads/main/docs/source/_static/img/tgp-logo.svg" width="20px" align="center" style="display: inline-block; height: 1.0em; width: unset; vertical-align: text-top;"/> tgp, we recommend browsing the [Documentation](https://torch-geometric-pool.readthedocs.io/en/latest/) to familiarize yourself with the API.

If you prefer a notebook-based introduction, check out the following tutorials:

* [![nbviewer](https://img.shields.io/badge/-Introduction-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/tgp-team/torch-geometric-pool/blob/main/docs/source/tutorials/intro.ipynb) Basic usage of common pooling operators, including how to pass arguments via aliases and inspect intermediate outputs.

* [![nbviewer](https://img.shields.io/badge/-Preprocessing-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/tgp-team/torch-geometric-pool/blob/main/docs/source/tutorials/preprocessing_and_transforms.ipynb) Demonstrates how to apply precomputed pooling methods and associated data transforms for faster training.

* [![nbviewer](https://img.shields.io/badge/-Advanced-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/tgp-team/torch-geometric-pool/blob/main/docs/source/tutorials/advanced.ipynb) Deep dive into the SRC framework, showing how each component interacts and how to use the select, reduce, connect and lift operations to modify the graph topology and the graph features.

In addition, check the [example folder](https://github.com/tgp-team/torch-geometric-pool/tree/main/examples) for a collection of minimalistic python script showcasing the usage of the pooling operators of <img src="https://raw.githubusercontent.com/tgp-team/torch-geometric-pool/refs/heads/main/docs/source/_static/img/tgp-logo.svg" width="20px" align="center" style="display: inline-block; height: 1.0em; width: unset; vertical-align: text-top;"/> tgp in all the most common downstream tasks, such as graph classification/regression, node classification/regression, and node clustering.

## Installation

<img src="https://raw.githubusercontent.com/tgp-team/torch-geometric-pool/refs/heads/main/docs/source/_static/img/tgp-logo.svg" width="20px" align="center" style="display: inline-block; height: 1.0em; width: unset; vertical-align: text-top;"/> tgp is compatible with Python>=3.9. We recommend installation
on a [Anaconda or Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install)
environment or a [virtual env](https://docs.python.org/3/library/venv.html).
<img src="https://raw.githubusercontent.com/tgp-team/torch-geometric-pool/refs/heads/main/docs/source/_static/img/tgp-logo.svg" width="20px" align="center" style="display: inline-block; height: 1.0em; width: unset; vertical-align: text-top;"/> tgp is conveniently available as a Python package on PyPI and 
can be easily installed using pip.

```bash
pip install torch-geometric-pool
```

For the latest version, consider installing from source:

```bash
pip install git+https://github.com/tgp-team/torch-geometric-pool.git
```

> [!CAUTION]
> <img src="https://raw.githubusercontent.com/tgp-team/torch-geometric-pool/refs/heads/main/docs/source/_static/img/tgp-logo.svg" width="20px" align="center" style="display: inline-block; height: 1.0em; width: unset; vertical-align: text-top;"/> tgp is built upon [PyTorch>=1.8](https://pytorch.org/) and [PyG>=2.6](https://github.com/pyg-team/pytorch_geometric/). Make sure you have both installed in your environment before installation. Check the [installation guide](https://torch-geometric-pool.readthedocs.io/en/latest/content/quickstart.html) for more details.

## Quick Example

```python
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
```

For more detailed examples, refer to the [Tutorials](https://torch-geometric-pool.readthedocs.io/en/latest/tutorials/index.html) and the coding [Examples](https://github.com/tgp-team/torch-geometric-pool/tree/main/examples).

## Contributing

Contributions are welcome! For major changes or new features, please open an issue first to discuss your ideas. See the [Contributing guidelines](./CONTRIBUTE.md) for more details on how to get involved. 
Help us build a better <img src="https://raw.githubusercontent.com/tgp-team/torch-geometric-pool/refs/heads/main/docs/source/_static/img/tgp-logo.svg" width="20px" align="center" style="display: inline-block; height: 1.0em; width: unset; vertical-align: text-top;"/> tgp!

Thanks to all contributors 🤝

<a href="https://github.com/tgp-team/torch-geometric-pool/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=tgp-team/torch-geometric-pool" />
</a>

## Citing

If you use Torch Geometric Pool for your research, please consider citing the library

```bibtex
@software{Bianchi_Torch_Geometric_Pool_2025,
    author = {Bianchi, Filippo Maria and Marisca, Ivan and Abate, Carlo},
    license = {MIT},
    month = {3},
    title = {{Torch Geometric Pool}},
    url = {https://github.com/tgp-team/torch-geometric-pool},
    year = {2025}
}
```

By [Filippo Maria Bianchi](https://sites.google.com/view/filippombianchi/home), [Ivan Marisca](https://marshka.github.io/) and [Carlo Abate](https://carloabate.it/).

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](https://github.com/tgp-team/torch-geometric-pool/blob/main/LICENSE) file for details.
