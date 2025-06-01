from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="buddy-ai",
    version="0.1.0",
    author="Your Name / AI",
    author_email="you@example.com",
    description="An AI agent to help with system administration and software development tasks.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/buddy-ai", # Replace with your actual URL if you have one
    packages=find_packages(exclude=['tests*', '.venv']),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'buddy=buddy_ai.cli:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Assuming MIT, update if different
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8', # Specify your Python version compatibility
)
