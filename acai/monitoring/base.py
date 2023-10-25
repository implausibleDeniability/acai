class MonitoringCallbackBase:
    def __call__(self, trainer):
        raise NotImplementedError()