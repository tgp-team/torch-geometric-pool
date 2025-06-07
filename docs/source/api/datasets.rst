.. py:module:: tgp.datasets
.. currentmodule:: tgp.datasets

Datasets
========

.. autosummary::
    :nosignatures:
    {% for cls in tgp.datasets.dataset_classes %}
        {{ cls }}
    {% endfor %}

{% for cls in tgp.datasets.dataset_classes %}
.. autoclass:: {{ cls }}
    :members:
{% endfor %}
