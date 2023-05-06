import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch",
    version="0.0.1",
    author="sunguoying",
    author_email="sunguoying21@gmail.com",
    description="coding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SGuoying/pytorch",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        # "jax",
        # "equinox",
        # "optax",
        # "numpy",
        # "transformers",
        # "datasets",
        # "tokenizers",
        # "torch >=1.8",
        # "pytorch_lightning",
        # "tensorflow",
        # "tensorflow_datasets"
        # "einops >=0.3",
        # "sentencepiece"
    ]
)
