from setuptools import find_packages, setup

setup(
    name="exrl",
    packages=find_packages(),
    version="1.4.25",
    description="Ex-RL Library",
    author="Ali Ghandi, Azam Kamranian, Mahyar Riazati",
    license="MIT",
    install_requires=[
        "gym",
        "tqdm",
        "stqdm",
        "scipy",
        "numpy",
        "pandas",
        "matplotlib",
        "hmmlearn>=0.2.8",
        "streamlit",
        "seaborn",
    ],
)
