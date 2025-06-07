.. py:module:: tgp.select
.. currentmodule:: tgp.select

Select
======

The most general interface is :class:`~tgp.select.Select`, which is the
parent class for every dataset in :mod:`tgp`.

.. autosummary::
    :nosignatures:
    {% for cls in tgp.select.select_classes %}
        {{ cls }}
    {% endfor %}

{% for cls in tgp.select.select_classes %}
.. autoclass:: {{ cls }}
    :members:
{% endfor %}
