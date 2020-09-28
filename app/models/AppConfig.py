class AppConfig(object):
    """
    Holding applications configuration as properties
    """
    def __init__(self):
        self.LoggingEnabled=False
        self.QueueSize=2
        self.NumWorkers=2