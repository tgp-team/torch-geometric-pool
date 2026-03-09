.. py:module:: tgp.poolers
.. currentmodule:: tgp.poolers

Poolers
=======

.. autosummary::
    :nosignatures:

    get_pooler
    {% for cls in tgp.poolers.pooler_classes %}
        {{ cls }}
    {% endfor %}

.. autofunction:: get_pooler

{% for cls in tgp.poolers.pooler_classes %}
.. autoclass:: {{ cls }}
    :members:
{% endfor %}
