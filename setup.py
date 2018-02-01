# this is only necessary when not using setuptools/distribute
from distutils.core import setup
from sphinx.setup_command import BuildDoc
cmdclass = {'build_doc': BuildDoc}

name = 'Sebastian Semper'
version = '0.1'
release = '0.1.0'

setup(
    name=name,
    author='Sebastian Semper',
    version=release,
    cmdclass=cmdclass,
    # these are optional and override conf.py settings
    command_options={
        'build_doc': {

        }}
    )
