from setuptools import setup, find_packages


def parse_requirements():
    with open('requirements.txt') as f:
        req = f.read().splitlines()
    return req


setup(
    name='dash_xas_tim',
    version="0.1.0",
    author='Jun Zhu',
    author_email='cas-support@xfel.eu',
    description='Xray absorption spectroscopy (XAS) data analysis with TIM',
    long_description='Offline data analysis and visualization tool for '
                     'Xray absorption spectroscopy (XAS) experiment with TIM'
                     'at European XFEL.',
    url='',
    license="BSD-3-Clause",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
        ],
    },
    package_data={},
    install_requires=parse_requirements(),
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
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
