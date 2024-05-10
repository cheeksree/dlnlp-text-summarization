from setuptools import setup

setup(name="dlnlp", version="1.0",
      description="NLP deep learning examples",
      long_description="NLP deep learning examples",
      url="https://github.com/cheeksree/dlnlp/",
      author="sree",
      author_email="cheeksree@gmail.com",
      packages=["dlnlp","app","Dataloaders","mlm"],
      license="MIT",
      requires=["numpy"])