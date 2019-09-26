from importlib import import_module

def split_package_module(full_path):
    components = full_path.split('.')
    package = components[0]
    module = '.'.join(components[1:])
    return package,module

def split_module_package(full_path):
    components = full_path.split('.')
    package = components[0]
    module = '.'.join(components[1:])
    return module,package

def split_module_class(full_path):
    components = full_path.split('.')
    class_name = components[-1]
    module_name = '.'.join(components[:-1])
    return class_name,module_name

def import_class(full_path):
    class_name,module_name = split_module_class(full_path)
    return getattr(import_module(module_name), class_name)
