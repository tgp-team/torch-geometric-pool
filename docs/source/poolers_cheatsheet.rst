:orphan:

Poolers Cheatsheet
==================

This cheatsheet provides a quick reference for the pooling operators available in :tgp:`tgp`.
It shows each pooler's **alias** (for :func:`tgp.poolers.get_pooler`), whether it supports sparse operations, has trainable parameters, and provides auxiliary losses.

* **sparse**: If checked (✓), the pooler operates on sparse adjacency matrices and returns sparse connectivity and :math:`\mathbf{S}` matrices, *e.g.*, suitable for sparse GNN layers like `GCNConv <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html>`__.
* **trainable**: If checked (✓), the pooler contains trainable parameters that are updated during training.
* **aux_loss**: If checked (✓), the pooler computes auxiliary loss terms that can be used for regularization during training.

Graph Pooling Operators
-----------------------

.. list-table::
    :widths: 15 35 10 10 10
    :header-rows: 1

    * - Alias
      - Class
      - sparse
      - trainable  
      - aux_loss
{% for pooler_name, class_name, sparse, trainable, aux_loss, paper_links in cheatsheet.get_pooler_cheatsheet() %}
    * - ``{{ pooler_name }}``
      - :class:`~tgp.poolers.{{ class_name }}`{% if paper_links %} [{% for paper_link in paper_links %}`{{ loop.index }} <{{ paper_link }}>`__{% if not loop.last %}, {% endif %}{% endfor %}]{% endif %}
      - {% if sparse %}✓{% endif %}
      - {% if trainable %}✓{% endif %}
      - {% if aux_loss %}✓{% endif %}
{% endfor %}