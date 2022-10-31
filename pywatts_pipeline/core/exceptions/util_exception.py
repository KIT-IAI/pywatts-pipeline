class UtilException(Exception):
    """
    Exception raised by a util function
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
