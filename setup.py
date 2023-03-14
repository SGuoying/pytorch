import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sunyata",
    version="0.0.1",
    author="fengwenfeng",
    author_email="fengwenfeng@gmail.com",
    description="coding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keepsimpler/sunyata",
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
