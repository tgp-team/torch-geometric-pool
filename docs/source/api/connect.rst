.. py:module:: tgp.connect
.. currentmodule:: tgp.connect

Connect
=======

.. autosummary::
    :nosignatures:
    {% for cls in tgp.connect.connect_classes %}
        {{ cls }}
    {% endfor %}

{% for cls in tgp.connect.connect_classes %}
.. autoclass:: {{ cls }}
    :members:
{% endfor %}
