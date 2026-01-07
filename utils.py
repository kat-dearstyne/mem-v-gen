import os

from dotenv import load_dotenv

load_dotenv()

def get_env_bool(name: str, default: bool = False) -> bool:
    """
    Get a boolean value from an environment variable.

    Args:
        name: The name of the environment variable
        default: The default value if the environment variable is not set

    Returns:
        Boolean value from the environment variable
    """
    default_str = str(default).lower()
    return os.getenv(name, default_str).lower() == "true"


def get_env_int(name: str, default: int = 0) -> int:
    """
    Get an integer value from an environment variable.

    Args:
        name: The name of the environment variable
        default: The default value if the environment variable is not set or invalid

    Returns:
        Integer value from the environment variable
    """
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_list(name: str, default: list[str] | None = None) -> list[str]:
    """
    Get a list of strings from a comma-separated environment variable.

    Args:
        name: The name of the environment variable
        default: The default value if the environment variable is not set

    Returns:
        List of strings from the environment variable
    """
    if default is None:
        default = []
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return [item.strip() for item in value.split(",")]
