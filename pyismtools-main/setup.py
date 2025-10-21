from setuptools import setup, find_packages

setup(
    name='pyismtools',
    version='0.4',
    description='Python package to perfom model/observation comparison with Meudon PDR models',
    url='https://gitlab.obspm.fr/ism/pyismtools',
    author='Emeric Bron',
    author_email='emeric.bron@obspm.fr',
    license='?',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=['tqdm>=4.64.1',
                      'numpy>=1.23.3',
                      'scipy>=1.9.1',
                      'matplotlib>=3.6.0',
                      'pymc>=5.13.1',
                      'pytensor>=2.20.0',
                      'arviz>=0.18.0'
                      ],

    classifiers=[
        'Development Status :: In progress',
        'Intended Audience :: Meudon PDR team',
    ],
)

