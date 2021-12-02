from setuptools import setup, Extension

def main():
    setup(name="cnets",
          version="1.0.0",
          description="C module for network computing",
          author="djanloo",
          ext_modules=[Extension("cnets", ["networkMDE/cnets.c", "networkMDE/cutils.c"])])

if __name__ == "__main__":
    main()
