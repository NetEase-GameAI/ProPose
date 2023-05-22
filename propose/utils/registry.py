import inspect


class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(name={self._name}, items={list(self._module_dict.keys())})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """ Register a module.

        Args:
            module (nn.Module): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError(f'module must be a class, but got {type(module_class)}')
        
        module_name = module_class.__name__
        
        if module_name in self._module_dict:
            raise KeyError(f'{module_name} is already registered in {self.name}')
        
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


def build_from_cfg(cfg, registry, default_args=None):
    """ Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (Registry): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'TYPE' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('TYPE')

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    return obj_cls(**args)
