from setuptools import setup,find_packages

setup(name="dlnlp", version="1.0",
      description="NLP deep learning Text Summarization ",
      long_description="Text Summarization",
      url="https://github.com/cheeksree/dlnlp/",
      author="sree",
      author_email="cheeksree@gmail.com",
      packages=find_packages(),
      license="MIT",
      install_requires=["numpy","torch","sklearn"])
