from setuptools import setup, find_packages

setup(
  name = 'dolg-tensorflow',
  packages = find_packages(exclude=['code_examples']),
  version = '0.0.2',
  license='MIT',
  description = 'Deep Orthogonal Fusion of Local and Global Features (DOLG) - TensorFlow 2 (Keras)',
  long_description=open('README.md').read(),
  long_description_content_type='text/markdown',
  author = 'M.Innat',
  author_email = 'innat1994@gmail.com',
  url = 'https://github.com/innat/DOLG-TensorFlow',
  keywords = [
    'deep learning',
    'image retrieval',
    'image recognition'
  ],
  install_requires=[
    'opencv-python>=4.1.2',
    'tensorflow>=2.7',
    'tensorflow-addons>=0.15.0'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 1 - Planning',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)