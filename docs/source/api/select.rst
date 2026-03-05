.. py:module:: tgp.select
.. currentmodule:: tgp.select

Select
======

The most general interface is :class:`~tgp.select.Select`, which is the
parent class for every :math:`\texttt{select}` operator in :mod:`tgp`.

.. autosummary::
    :nosignatures:
    {% for cls in tgp.select.select_classes %}
        {{ cls }}
    {% endfor %}
    {% for func in tgp.select.select_functions %}
        {{ func }}
    {% endfor %}

{% for cls in tgp.select.select_classes %}
.. autoclass:: {{ cls }}
    :members:
{% endfor %}

{% for func in tgp.select.select_functions %}
.. autofunction:: {{ func }}
{% endfor %}
