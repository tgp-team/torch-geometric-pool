.. py:module:: tgp.connect
.. currentmodule:: tgp.connect

Connect
=======

.. autosummary::
    :nosignatures:
    {% for cls in tgp.connect.connect_classes %}
        {{ cls }}
    {% endfor %}
    {% for func in tgp.connect.connect_functions %}
        {{ func }}
    {% endfor %}

{% for cls in tgp.connect.connect_classes %}
.. autoclass:: {{ cls }}
    :members:
{% endfor %}

{% for func in tgp.connect.connect_functions %}
.. autofunction:: {{ func }}
{% endfor %}
