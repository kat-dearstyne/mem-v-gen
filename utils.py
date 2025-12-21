import os


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
