import functools
import json
from typing import Any, Callable, Union
from .form import SubmitForm


def use_kwargs(fields: dict,
               locations: Union[str, tuple] = None,
               _rasie: bool = True,
               ) -> Callable[..., Any]:
    def wrapper(func):
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any):
            form = SubmitForm(**kwargs)
            data, error = form.bind(args[0])
            if error:
                if _rasie:
                    raise ValueError(json.dumps(error))
            kwargs.update(data)
            await func(*args, **kwargs)
        return wrapper
    return wrapper
