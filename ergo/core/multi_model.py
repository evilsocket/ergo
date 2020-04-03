
class multi_model():
    def __init__ (self, cpu_model, gpu_model = None):
        self.cpu_model = cpu_model
        self.gpu_model = gpu_model

    def fit (self, x=x, y=y, **kwargs):
        return self.gpu_model.fit(x, y, **kwargs)

    def compile(self, **kwargs):
        return self.gpu_model.compile(**kwargs)

    def __getattr__(self, name, *args, **kwargs):
        if hasattr(self.cpu_model, name):
            def wrapper(*args, **kwargs):
                return getattr(self.cpu_model, name)(*args, **kwargs)
            return wrapper
        raise AttributeError("Method/Attribute %s not found for keras model" % name)
