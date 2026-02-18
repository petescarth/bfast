import os
import shutil
from setuptools import setup, find_packages, Command

DISTNAME = 'bfast'
DESCRIPTION = 'A Python library for Breaks For Additive Season and Trend (BFAST) that resorts to parallel computing for accelerating the computations.'
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Dmitry Serykh'
MAINTAINER_EMAIL = 'dmitry.serykh@gmail.com'
URL = 'https://bfast.readthedocs.io'
LICENSE = 'GNU GENERAL PUBLIC LICENSE'
DOWNLOAD_URL = 'https://github.com/petescarth/bfast'

# Read version directly or import if safe. 
# Importing bfast might fail if dependencies are not installed, so let's try to find it or just assume it is in the package.
# For now, I'll hardcode it or read it from __init__.py if I could, but let's try to import it inside a try block or similar.
# Actually, the original imported it. Let's try to extract it from bfast/__init__.py to avoid import errors during setup.
def get_version():
    with open(os.path.join('bfast', '__init__.py')) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip("'").strip('"')
    return '0.0.0'

VERSION = get_version()

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        if os.path.exists('build'):
            shutil.rmtree('build')
        if os.path.exists('dist'):
            shutil.rmtree('dist')
        if os.path.exists('bfast.egg-info'):
            shutil.rmtree('bfast.egg-info')
        
        for dirpath, dirnames, filenames in os.walk('bfast'):
            for filename in filenames:
                if (filename.endswith('.so') or \
                    filename.endswith('.pyd') or \
                    filename.endswith('.dll') or \
                    filename.endswith('.pyc') or \
                    filename.endswith('_wrap.c') or \
                    filename.startswith('wrapper_') or \
                    filename.endswith('~')):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))

setup(
    name=DISTNAME,
    version=VERSION,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/x-rst',
    license=LICENSE,
    url=URL,
    download_url=DOWNLOAD_URL,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'numpy>=1.24',
        'pandas>=2.0',
        'pyopencl>=2022.1',
        'scikit-learn>=1.0',
        'scipy>=1.10',
        'matplotlib>=3.7',
        'wget>=3.2',
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    cmdclass={'clean': CleanCommand},
    python_requires='>=3.8',
)
