"""Reference validation behavior of the installed pydantic version."""

from pydantic import ConfigDict, ValidationError, validate_call


def validation_title(qualname: str) -> str:
    """Read the title from pydantic directly, independently of the fse wrapper."""

    def entry(_value: int) -> None:
        pass

    entry.__name__ = qualname.rsplit(".", 1)[-1]
    entry.__qualname__ = qualname
    validated = validate_call(config=ConfigDict(strict=True))(entry)
    try:
        validated("1")
    except ValidationError as error:
        return error.title
    raise AssertionError("pydantic accepted a string for a strict integer")
