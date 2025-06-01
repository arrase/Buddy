from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='buddyai',
    version='0.1.0',
    author='AI Agent', # Generic author
    author_email='buddyai@example.com', # Placeholder email
    description='An AI agent for planning and executing tasks, including shell commands.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/example/buddy-agent', # Placeholder URL
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'buddyai=buddy_agent.cli.main:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Assuming MIT
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Shells',
    ],
    python_requires='>=3.8', # Updated to a common modern version, can be adjusted
)
