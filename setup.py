#!/usr/bin/env python
try:
    from numpy.distutils.core import Extension as NumpyExtension
    from numpy.distutils.core import setup

    from distutils.extension import Extension
    from Cython.Build import cythonize

    import numpy

except ImportError:
    raise ImportError('Numpy needs to be installed or updated.')


extensions = [
    NumpyExtension(
        name='BayHunter.surfdisp96_ext',
        sources=['src/extensions/surfdisp96.f'],
        extra_f77_compile_args='-O3 -ffixed-line-length-none -fbounds-check -m64'.split(),  # noqa
        f2py_options=['only:', 'surfdisp96', ':'],
        language='f77'),
    ]

extensions.extend(cythonize(
    Extension("BayHunter.rfmini", [
        "src/extensions/rfmini/rfmini.pyx",
        "src/extensions/rfmini/greens.cpp",
        "src/extensions/rfmini/model.cpp",
        "src/extensions/rfmini/pd.cpp",
        "src/extensions/rfmini/synrf.cpp",
        "src/extensions/rfmini/wrap.cpp",
        "src/extensions/rfmini/fork.cpp"],
        include_dirs=[numpy.get_include()])))

# sources = [
#     "src/extensions/raysum_src/eispack-cg.f",
#     "src/extensions/raysum_src/matrixops.f",
#     "src/extensions/raysum_src/eigenvec.f",
#     "src/extensions/raysum_src/raysum.f",
#     "src/extensions/raysum_src/readwrite.f",
#     # "src/extensions/raysum_src/params.h"
#     "src/extensions/raysum_src/phaselist.f",
#     "src/extensions/raysum_src/buildmodel.f",
#     "src/extensions/raysum_src/trace.f",
#     # "src/extensions/raysum_src/misfit.f",
#     "src/extensions/raysum_src/seis-spread.f"
# ]

# extensions.extend(
#     NumpyExtension(
#         name="BayHunter.raysum_ext",
#         sources=sources,
#         extra_f77_compile_args=["-Wall", "-O3", "-g"],
#         f2py_options=["only:", "mcmcraysum", ":"],
#         language="f77"),
# )

setup(
    name="BayHunter",
    version="2.1",
    author="Jennifer Dreiling",
    author_email="jennifer.dreiling@gfz-potsdam.de",
    description=("Transdimensional Bayesian Inversion of RF and/or SWD."),
    install_requires=[],
    url="https://github.com/jenndrei/BayHunter",
    packages=['BayHunter'],
    package_dir={
        'BayHunter': 'src'},

    scripts=['src/scripts/baywatch'],

    package_data={
        'BayHunter': ['defaults/*'], },

    ext_modules=extensions
)
