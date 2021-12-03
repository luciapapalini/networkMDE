from setuptools import setup, Extension

with open('requirements.txt', 'r') as reqfile:
    dependencies = reqfile.readlines()
print(dependencies)

def main():
    setup(name="cnets",
          version="1.0.0",
          description="C module for network computing",
          author="djanloo",
          ext_modules=[Extension("cnets", ["networkMDE/cnets.c", "networkMDE/cutils.c"])],
          install_requires=dependencies)

if __name__ == "__main__":
    main()
