# TIMprocess

Topological Image Modification (TIM) and Topological Image Processing (TIP). 

Main paper this repository is based on: https://www.nature.com/articles/s41598-020-77933-y

# Info

To start:
- Clone/download the repository.
- Install the dependencies below.
- Open the notebook "TIP_tutorial.ipynb".

Or:
- You can preview the notebook in github.


# Dependencies

Most of these can be installed directly through the following command line: $pip install <package_name> 

Package dependencies:
- "numpy" (https://numpy.org/)
- "matplotlib" (https://matplotlib.org/)
- "Pillow" (https://pillow.readthedocs.io/en/stable/)
- "scikit-image" (https://scikit-image.org/)
- "OpenCV" (https://pypi.org/project/opencv-python/)
- "SciPy" (https://www.scipy.org/)
- "ripser" (https://pypi.org/project/ripser/)

# Suggests

The current implementation computes 0-dimensional persistent homology through the Python library ripser. This implementation does not allow one to match diagram points to birth/death pixels. We therefore require all pixel values to be unique, as illustrated on https://ripser.scikit-tda.org/en/latest/notebooks/Lower%20Star%20Image%20Filtrations.html. We observed that this may rarely lead to wrong matchings, due to small numerical errors. To overcome this, we recommend looking into different implementations that may allow one to track birth/death pixels, e.g., https://pypi.org/project/dionysus/.


# Contact

Robin.Vandaele@UGent.be
