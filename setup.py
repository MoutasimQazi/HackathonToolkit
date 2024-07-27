from setuptools import setup, find_packages
 
classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
]

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('CHANGELOG.txt', 'r', encoding='utf-8') as f:
    long_description += '\n\n' + f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
  name='HackathonToolkit',
  version='0.0.6',
  description='The Hackathon Toolkit simplifies hackathon projects with utilities for setup, API interaction, data handling, ML, visualization, collaboration, and deployment, automating tasks for innovation focus.',
  long_description=long_description,
  long_description_content_type='text/markdown',  
  url='', 
  author='Moutasim Qazi',
  author_email='moutasimqazi@gmail.com',
  license='MIT',
  classifiers=classifiers,
  keywords='hackathon , toolkit , python , library', 
  packages=find_packages(),
  install_requires=requirements
)
