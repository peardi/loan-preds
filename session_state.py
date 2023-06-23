class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key):
        return getattr(self, key, None)

    def set(self, key, value):
        setattr(self, key, value)
