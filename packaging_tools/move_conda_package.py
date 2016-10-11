import sys
import os
import glob
import shutil

from conda_build.config import get_or_merge_config

packages_dir = get_or_merge_config(None).bldpkgs_dir

binary_package_glob = os.path.join(packages_dir, 'spyking-circus*.tar.bz2')
binary_packages = glob.glob(binary_package_glob)
for binary_package in binary_packages:
    shutil.move(binary_package, '.')
