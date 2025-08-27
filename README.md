# eta-ai
AI that predicts ETA from distance and traffic level

Idea Generation:

have a data set of travel time, traffic level (1,2,3), and distance (km)
get chatgpt to make training csv data set

use a linera regression model to train

resources to research:
- kaggle - intro to ml




- statquest - linear reg





- scikit - linear reg (LR) documentaiotn
    class sklearn.linear_model.LinearRegression(*, fit_intercept=True, copy_X=True, tol=1e-06, n_jobs=None, positive=False)
    ordinary least squares LR

    useful params:
        fit_intercept:
            distance = 0 must mean eta = 0
        positive:
            useful for inforcing non-negative coefficeints 


    import numpy as np
    from sklearn.linear_model import LinearRegression

    ## IMPORTS


    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # INPUT DATA
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3


    reg = LinearRegression().fit(X, y)
    # TRAIN THE MODEL, CREATED LR OBJECT

    reg.score(X, y)
    >>> 1.0
    reg.coef_
    >>> array([1., 2.])
    reg.intercept_
    >>> np.float64(3.0)
    reg.predict(np.array([[3, 5]]))
    >>> array([16.])
    # for x_0 = 3, x_1 = 5 model predicts 16


    predict(X)

    --> this means we need to take the data as an array
    

    set_params(**params)
    --> **params : dict



- fastapi serving model


train -> save model -> reload later -> predict ETA for new inputs

wrap model with small web service (fast api):
|--> so i can do something like 'POST {distance:20, traffic: 2} ----> {ETA: 35 mins documentaiotn}
