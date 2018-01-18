### About

A simplified collection of a time-series analysis methods.

### Methods Collection

* [Autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation) <- wiki
    - Autocorrelation, also known as serial correlation, is the correlation of a signal with a delayed copy of itself as a function of delay. Informally, it is the similarity between observations as a function of the time lag between them.
* [Cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation) <- wiki
    - In signal processing, cross-correlation is a measure of similarity of two series as a function of the displacement of one relative to the other. This is also known as a sliding dot product or sliding inner-product.
* [Autoregressive–moving-average model](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model) <- wiki
    - In the statistical analysis of time series, autoregressive–moving-average (ARMA) models provide a parsimonious description of a (weakly) stationary stochastic process in terms of two polynomials, one for the autoregression and the second for the moving average.
    - [Box–Jenkins method](https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method) <- wiki
        + In time series analysis, the Box–Jenkins method, applies autoregressive moving average (ARMA) or autoregressive integrated moving average (ARIMA) models to find the best fit of a time-series model to past values of a time series.
* [Autoregressive conditional heteroskedasticity (ARCH)](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity) <- wiki
    - The autoregressive conditional heteroskedasticity (ARCH) model is a statistical model for time series data that describes the variance of the current error term or innovation as a function of the actual sizes of the previous time periods' error terms; often the variance is related to the squares of the previous innovations.
    - ARCH models are commonly employed in modeling financial time series that exhibit time-varying volatility clustering, i.e. periods of swings interspersed with periods of relative calm.
* [Vector autoregression](https://en.wikipedia.org/wiki/Vector_autoregression) <- wiki
    - Vector autoregression (VAR) is a stochastic process model used to capture the linear interdependencies among multiple time series. VAR models generalize the univariate autoregressive model (AR model) by allowing for more than one evolving variable.
* [What is Exponential Smoothing?](http://www.itl.nist.gov/div898/handbook/pmc/section4/pmc43.htm)
    - This is a very popular scheme to produce a smoothed Time Series. Whereas in Single Moving Averages the past observations are weighted equally, Exponential Smoothing assigns exponentially decreasing weights as the observation get older.
    - [Forecasting with Single Exponential Smoothing](http://www.itl.nist.gov/div898/handbook/pmc/section4/pmc43.htm)
    - NOTE: *DOESN'T ADDRESS DATA REQUIREMENTS*
* [Spectral Analysis](https://faculty.washington.edu/dbp/PDFFILES/GHS-AP-Stat-talk.pdf) <- PDF
    - A method that tries to describe a particular time series using sines and consines (e.g. Fourier representation)
    - Allows to reexpress a time series in standard way and anables an analysis between different time series.
    - [Basic Singular Spectrum Analysis and Forecasting with R](https://arxiv.org/pdf/1206.6910.pdf)
    - [Singular Spectrum Analysis for time series forecasting in Python with Jupyter Notebook](https://github.com/aj-cloete/pySSA/blob/master/Singular%20Spectrum%20Analysis%20Example.ipynb)
        + NOTE: *PROBABLY DOESN'T ADDRESS PMML REQUIREMENTS*

### Tools and Implementations Collection

* PMML
    - [PMML 4.3 - Time Series Models](http://dmg.org/pmml/v4-3/TimeSeriesModel.html)
        + In looks like in PMML 4.3, only **Exponential Smoothing** is defined, whether other algorithms/methods are planned for later versions and have only placeholders
        + It looks like situation didn't change since 2010 -> http://standardwisdom.com/softwarejournal/2010/11/arima-and-seasonality-adjustment-support-in-pmml-4-0/
        + Was suggested in 2011, updated in 2016, maybe will appear in 4.4 PMML version (according to the issue tracker of DMG - http://mantis.dmg.org/)
        + PMML Example - http://dmg.org/pmml/pmml_examples/index.html
    - [JPMML-SkLearn](https://github.com/jpmml/jpmml-sklearn)
* [ARCH models in Python](https://github.com/bashtage/arch#volatility) <- github
    - [ARCH Modeling, An example with IPython Notebook](http://nbviewer.jupyter.org/github/bashtage/arch/blob/master/examples/univariate_volatility_modeling.ipynb)


### "Methods and Code" Examples

* [How to Decompose Time Series Data into Trend and Seasonality](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/)
* [A comprehensive beginner’s guide to create a Time Series Forecast (with Codes in Python)](https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/)
* [How to Create an ARIMA Model for Time Series Forecasting with Python](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)
