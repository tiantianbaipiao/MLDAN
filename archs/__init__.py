import importlib
from os import path as osp

from basicsr.utils import scandir


# Get the absolute path of the current file's directory
arch_folder = osp.dirname(osp.abspath(__file__))

# List comprehension to extract base names (without extensions) of files ending with '_arch.py'
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]

# Dynamically import modules that match the naming convention (end with '_arch.py')
_arch_modules = [importlib.import_module(f'archs.{file_name}') for file_name in arch_filenames]

