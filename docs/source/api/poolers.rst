.. py:module:: tgp.poolers
.. currentmodule:: tgp.poolers

Poolers
=======

.. autosummary::
    :nosignatures:
    {% for cls in tgp.poolers.pooler_classes %}
        {{ cls }}
    {% endfor %}

{% for cls in tgp.poolers.pooler_classes %}
.. autoclass:: {{ cls }}
    :members:
{% endfor %}
