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

def hierarchical_sort(lst, direction='lr', order='descending'):
    assert len(lst) > 0
    to_sort = lst[:]
    sample = to_sort[0]
    sort_order = -1 if order == 'descending' else 1
    if direction == 'rl':
        for i in range(len(sample)):
            to_sort.sort(key=lambda x: sort_order * x[i])
    elif direction == 'lr':
        for j in range(len(sample)-1, -1, -1):
            to_sort.sort(key=lambda x: sort_order * x[j])
    return to_sort


