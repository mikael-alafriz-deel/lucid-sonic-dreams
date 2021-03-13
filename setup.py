import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open('requires.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="lucidsonicdreams", 
    version="0.3",
    author="Alain Mikael Alafriz",
    author_email="mikaelalafriz@gmail.com",
    description="Syncs GAN-generated visuals to music",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mikaelalafriz/lucid-sonic-dreams",
    download_url="https://github.com/mikaelalafriz/lucid-sonic-dreams/archive/v_03.tar.gz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required
)
