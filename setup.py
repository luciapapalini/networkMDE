from setuptools import setup, Extension

with open('requirements.txt', 'r') as reqfile:
    dependencies = reqfile.readlines()

def main():
    setup(name="networkMDE",
          version="1.0.0",
          packages=["networkMDE"],
          description="Module for network computing",
          author="djanloo",
          author_email='becuzzigianluca@gmail.com',
          ext_modules=[Extension("cnets", ["networkMDE/cnets.c", "networkMDE/cutils.c"])],
          install_requires=dependencies)

if __name__ == "__main__":
    main()
