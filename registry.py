class Registry(object):
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def inner_wrapper(wrapped_class):
            self._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    def get(self, name):
        return self._registry[name]


# 创建一个全局的注册表实例
registry = Registry()