from setuptools import setup, find_packages

def load_requirements(f):
    retval = [str(k.strip()) for k in open(f, "rt")]
    return [k for k in retval if k and k[0] not in ("#", "-")]

try:
    long_desc = open("README.rst").read()
except IOError:
    long_desc = "Failed to read README.rst"

setup(
    name="m05",
    version="1.2.13",

    description="Project on reproductibility in science",

    url="https://github.com/Laurasid/M05_Project",

    license="MIT",
    author="Laura Sidler, Jerome Amos",
    author_email="sidler@icare.ch, amos@icare.ch",

    long_description=long_desc,
    long_description_content_type="text/x-rst",

    packages=find_packages(),
    include_package_data=True,

    install_requires=load_requirements("requirements.txt"),

    entry_points={"console_scripts": ["m05-run = src.main:main"]},

    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
