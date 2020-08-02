from distutils.core import setup

setup(
    name='LittleThings',
    version='0.1.0',
    author='Anna Kwa',
    author_email='anna.s.kwa@gmail.com',
    packages=['little_things'],
    license='LICENSE.txt',
    description='Model galaxy 1st moment maps with MCMC packages.',
    install_requires=[
        "astropy",
        "emcee==2.2.1",
        "missingpy",
        "numpy",
        "scipy",
    ],
)
