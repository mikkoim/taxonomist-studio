from setuptools import setup, find_packages

setup(
    name='taxonomist_studio',
    version='0.1.0',
    packages=find_packages(include=['taxonomist_studio']),
    package_dir={'':'src'},
    entry_points={
        "console_scripts": [
            "taxonomist_studio=taxonomist_studio:main",
            "taxonomist-studio=taxonomist_studio:main"
        ]
    }
)