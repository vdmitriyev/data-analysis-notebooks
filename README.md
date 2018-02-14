### About

Collection of the different sort of own data analysis performed with help of IPython and it's amazing Notebook (now known as [Jupyter Notebook](http://jupyter.org/)).

### Dependencies

* Install one of the following
    - [Python 2.7](http://www.python.org/download/) + Packages
    - [Anacoda](https://www.continuum.io/downloads)
* Update Anaconda
    - [How to update already installed anaconda](https://stackoverflow.com/questions/45197777/how-do-i-update-anaconda)
    - ```
        conda update conda
        conda update anaconda
        conda update --all
      ```

### Data Analysis

* **Plotting data from the weather web site [tutiempo.net](http://en.tutiempo.net/)**
    - Check this analysis at the [nbviewer](http://nbviewer.ipython.org/github/vdmitriyev/data-analysis-ipython/blob/master/tutiempo/tutiempo.ipynb) or at [github](https://github.com/vdmitriyev/data-analysis-ipython/blob/master/tutiempo/tutiempo.ipynb)
* **Sample analysis of Iris with sklearn**
    - Check this analysis at the [nbviewer](http://nbviewer.ipython.org/github/vdmitriyev/data-analysis-ipython/blob/master/iris/iris.ipynb) or at [github](https://github.com/vdmitriyev/data-analysis-ipython/blob/master/iris/iris.ipynb)
* **Repeating data analysis of energy efficiency dataset previouslt done in publication**
    - Check
    - Based on publication ['Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools'](https://scholar.google.com/scholar?hl=en&q=A.+Tsanas%2C+A.+Xifara%3A+%27Accurate+quantitative+estimation+of+energy+performance+of+residential+buildings+using+statistical+machine+learning+tools%27%2C+Energy+and+Buildings%2C+Vol.+49%2C+pp.+560-567%2C+2012&btnG=&as_sdt=1%2C5&as_sdtp=) by A. Tsanas and A. Xifara published in Energy and Buildings, Vol. 49, pp. 560-567, 2012.
* **Plotting irradiance data gathered by Solar Measurement Grid, Oahu, Hawaii provided by NREL**
    - To download data you may want to use script "download_nrel_oahu_solar.py" provided inside "nrel-oahu-solar/data" folder

### Starting

* Start Jupyther Notebook with Anacoda
```bash
jupyter notebook
```
* Start IPython with Anaconda on machine with Windows for github repository
```bash
activate analysis
ipython notebook --ip='*' --notebook-dir=D:\git\data-analysis-ipython\
```

### Data Analytics Materials

* Data Preprocessing
    + [Preprocessing in Data Science (Part 1): Centering, Scaling, and knn](https://www.datacamp.com/community/tutorials/preprocessing-in-data-science-part-1-centering-scaling-and-knn#gs.XJ7SfLk)
    + [Kaggle bulldozers: Basic cleaning](http://danielfrg.com/blog/2013/03/07/kaggle-bulldozers-basic-cleaning/)

* Jupyter Notebooks from Books
    + [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook) by Jake Vanderplas

* Tutorials and Videos
    + [Reproducible Data Analysis in Jupyter](https://jakevdp.github.io/blog/2017/03/03/reproducible-data-analysis-in-jupyter/)
        - Very short, but descriptive video colletion by Jake Vanderplas, git repository is here - [JupyterWorkflow](https://github.com/jakevdp/JupyterWorkflow)

### Author

* Viktor Dmitriyev
