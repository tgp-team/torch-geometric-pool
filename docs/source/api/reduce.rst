.. py:module:: tgp.reduce
.. currentmodule:: tgp.reduce

Reduce
======

The most general interface is :class:`~tgp.reduce.Reduce`, which is the
parent class for every reduce operator in :mod:`tgp`.

.. autosummary::
    :nosignatures:
    {% for cls in tgp.reduce.reduce_classes %}
        {{ cls }}
    {% endfor %}

{% for cls in tgp.reduce.reduce_classes %}
.. autoclass:: {{ cls }}
    :members:
{% endfor %}

{% for func in tgp.reduce.reduce_functions %}
.. autodata:: {{ func }}
   :annotation:
{% endfor %}