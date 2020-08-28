import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='ESRGAN-pytorch-qap',
    version='0.0.1',
    author='Jack Yu',
    author_email='jackchungchiehyu@gmail.com',
    description='ESRGAN for Pytorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/qAp/ESRGAN-pytorch',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6')
