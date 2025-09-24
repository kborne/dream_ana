from setuptools import setup, find_packages
  
setup(
    name="dream",
    version="0.1.0",
    packages=find_packages(),            # finds dream, dream.alg, dream.util, dream.lib, etc
    include_package_data=True,           # include package_data in wheels
    package_data={
        # bundle your native extension under dream/lib
        "dream.lib": ["*.so"],
    },
    install_requires=[
        # e.g. "numpy>=1.20",
    ],
    entry_points={
        "console_scripts": [
            # installs a dream executable that calls dream.main:main()
            "dream = dream.cli:run",
        ],
    },
    zip_safe=False,  # ensures .so files load correctly
)
