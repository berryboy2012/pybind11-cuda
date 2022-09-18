import platform
if platform.system() == 'Windows':
    # https://bugs.python.org/issue43173
    import os
    from pathlib import Path
    for path in os.environ['Path'].split(';'):
        if len(list(Path(path).glob('cudart*.dll'))) > 0:
            os.add_dll_directory(path)
            break
else:
    # Things should be taken care of by $LD_LIBRARY_PATH
    pass
from .basic_module import multiply_with_scalar
# wrap something over the CXX modules if you want here
__all__ = ['multiply_with_scalar']
