class ModelNotSupportedError(Exception):
    """
    Exception raised when an unsupported or incompatible model is requested.

    This error is typically thrown during model initialization when the
    provided model identifier or configuration is not supported by the
    system or current implementation.

    Parameters
    ----------
    message : str
        A descriptive message explaining why the model initialization failed.
    """

    def __init__(self, message: str):
        super().__init__(f"Failed to initialize model: {message}")