from setuptools import setup, find_packages
from sphinx.setup_command import BuildDoc

setup(
    name='camiba',
    version='0.1',
    release='0.1.0',
    author='Sebastian Semper',
    author_email='sebastian.semper@tu-ilmenau.de',
    license='LGPL3',
    install_requires=['sphinx', 'numpydoc'],
    description='Compressed Sensing Algorithms for Python',
    url='https://github.com/SebastianSemper/camiba',
    cmdclass={'build_doc': BuildDoc},
    # these are optional and override conf.py settings
    command_options={
        'build_doc': {
        }}
)
