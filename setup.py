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
with open('README.txt', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('CHANGELOG.txt', 'r', encoding='utf-8') as f:
    long_description += '\n\n' + f.read()

setup(
  name='HackathonToolkit',
  version='0.0.1-beta',
  description='The Hackathon Toolkit simplifies hackathon projects with utilities for setup, API interaction, data handling, ML, visualization, collaboration, and deployment, automating tasks for innovation focus.',
  long_description=long_description,
  long_description_content_type='text/plain',  # Adjust content type based on your actual format
  url='',  # Update with your project's URL
  author='Moutasim Qazi',
  author_email='moutasimqazi@gmail.com',
  license='Apache License',
  classifiers=classifiers,
  keywords='hackathon , toolkit , python , library', 
  packages=find_packages(),
  install_requires=[''] 
)
