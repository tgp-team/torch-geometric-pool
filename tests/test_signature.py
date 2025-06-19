import pytest

from tgp.utils.signature import foo_signature


def test_foo_signature_with_no_args():
    # Define a function that takes no parameters
    def no_args():
        pass

    sig = foo_signature(no_args)
    # Expect that args = [], and both flags are False
    assert sig.args == []
    assert sig.has_varargs is False
    assert sig.has_kwargs is False


def test_foo_signature_first_arg_not_self_or_cls():
    # Define a function whose first parameter is "x" (not "self" or "cls")
    def foo(x, y, *args, **kwargs):
        pass

    sig = foo_signature(foo)
    # Since first arg is "x", the `if` should be skipped.
    # Therefore, sig.args should still be ["x", "y"] (and varargs/kwargs are True).
    assert sig.args == ["x", "y"]
    assert sig.has_varargs is True
    assert sig.has_kwargs is True


if __name__ == "__main__":
    pytest.main([__file__])
