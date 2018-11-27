from setuptools import setup, find_packages


REQUIREMENTS = open('requirements.txt', encoding='utf-8').readlines()
REQUIREMENTS = [req.rstrip() for req in REQUIREMENTS]


setup(
    name='karaboXAS',
    version="0.1.0",
    author='Jun Zhu',
    author_email='jun.zhu@xfel.eu',
    description='Xray absorption spectroscopy (XAS) data analysis',
    long_description='Offline data analysis and visualization tool for '
                     'Xray absorption spectroscopy (XAS) experiment '
                     'at European XFEL.',
    url='',
    license="BSD-3-Clause",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
        ],
    },
    package_data={},
    install_requires=REQUIREMENTS,
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Physics',
    ]
)
