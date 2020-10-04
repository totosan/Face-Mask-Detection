class AppConfig(object):
    """
    Holding applications configuration as properties
    """
    def __init__(self):
        self.LoggingEnabled=False
        self.QueueSize=10
        self.NumWorkers=1

    def __str__(self):
        return (f'Logging: {str(self.LoggingEnabled)}\nQueueSize:{str(self.QueueSize)}\nWorker count: {str(self.NumWorkers)}')