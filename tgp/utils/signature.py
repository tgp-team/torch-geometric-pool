import inspect
from typing import Callable, NamedTuple, Type, Union


class Signature(NamedTuple):
    args: list
    has_varargs: bool
    has_kwargs: bool


def foo_signature(foo: Union[Callable, Type]) -> Signature:
    if isinstance(foo, type):
        foo = foo.__init__
    argspec = inspect.getfullargspec(foo)
    args = argspec.args
    if len(args) and args[0] in ["self", "cls"]:  # temp, to do better
        args = args[1:]
    has_varargs = argspec.varargs is not None
    has_kwargs = argspec.varkw is not None
    return Signature(args=args, has_varargs=has_varargs, has_kwargs=has_kwargs)
