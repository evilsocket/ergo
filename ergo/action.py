class Action(object):
    def __init__(self, name, description, cb):
        self.name = name
        self.description = description
        self.cb = cb
