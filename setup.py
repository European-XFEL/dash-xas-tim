from setuptools import setup, find_packages


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
    install_requires=[
        'numpy>=1.16',
        'pandas>=0.25',
        'dash>=1.1.0',
        'dash-daq>=0.1.7',
        'flask_caching>=1.7.2',
        'karabo_data>=0.6.2'
    ],
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
