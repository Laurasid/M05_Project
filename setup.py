from setuptools import setup, find_packages

def load_requirements(f):
    retval = [str(k.strip()) for k in open(f, "rt")]
    return [k for k in retval if k and k[0] not in ("#", "-")]

try:
    long_desc = open("README.rst").read()
except IOError:
    long_desc = "Failed to read README.rst"

setup(
    name="repro_m05",
    version="1.1.1",

    description="Project on reproductibility in science",

    url="https://github.com/Laurasid/M05_Project",

    license="MIT",
    author="Laura Sidler, Jerome Amos",

    long_description=long_desc,
    long_description_content_type="text/x-rst",

    packages=find_packages(),
    include_package_data=True,

    install_requires=load_requirements("requirements.txt"),

    entry_points={"console_scripts": ["run = src:main"]},

    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
