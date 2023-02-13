import weakref


class UniqueIDGenerator:
    """
    This class creates unique IDs for an object that is unique through
    the whole lifetime of the python application.
    """

    def __init__(self):
        self.id_to_obj = {}
        self.id_to_true_id = {}
        self.counter = 0

    def get_id(self, obj):
        self.counter += 1
        if (id(obj) not in self.id_to_obj) or self.id_to_obj[id(obj)]() is None:
            self.id_to_obj[id(obj)] = weakref.ref(obj)
            self.id_to_true_id[id(obj)] = self.counter
        return self.id_to_true_id[id(obj)]
