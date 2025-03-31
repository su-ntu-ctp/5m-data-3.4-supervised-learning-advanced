# Assignment Solution

## Instructions

1.  **Regularization**:

    - Use the `diabetes` dataset from `sklearn.datasets`.
    - Compare the performance (Mean Squared Error) of `LinearRegression`, `Ridge`, and `Lasso` models.
    - Tune the `alpha` parameter for `Ridge` and `Lasso` using `GridSearchCV` with cross-validation to find the optimal regularization strength.

    ```python
    from sklearn.datasets import load_diabetes

    # Load the diabetes dataset
    diabetes = load_diabetes()
    ```

    ```python
    import numpy as np
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import mean_squared_error

    # Prepare data
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Linear Regression
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    y_pred_linear = linear_reg.predict(X_test)
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    print(f'Linear Regression MSE: {mse_linear}')

    # Ridge Regression with GridSearchCV
    ridge = Ridge()
    parameters_ridge = {'alpha': np.logspace(-5, 5, 11)}
    ridge_regressor = GridSearchCV(ridge, parameters_ridge, scoring='neg_mean_squared_error', cv=5)
    ridge_regressor.fit(X_train, y_train)
    y_pred_ridge = ridge_regressor.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    print(f'Ridge Regression MSE: {mse_ridge}')
    print(f'Best alpha for Ridge: {ridge_regressor.best_params_}')

    # Lasso Regression with GridSearchCV
    lasso = Lasso()
    parameters_lasso = {'alpha': np.logspace(-5, 5, 11)}
    lasso_regressor = GridSearchCV(lasso, parameters_lasso, scoring='neg_mean_squared_error', cv=5)
    lasso_regressor.fit(X_train, y_train)
    y_pred_lasso = lasso_regressor.predict(X_test)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    print(f'Lasso Regression MSE: {mse_lasso}')
    print(f'Best alpha for Lasso: {lasso_regressor.best_params_}')
    ```

2.  **Ensemble Methods**:

    - Use the `breast_cancer` dataset from `sklearn.datasets`.
    - Compare the performance (F1 Score and AUC) of `DecisionTreeClassifier`, `RandomForestClassifier`, and `GradientBoostingClassifier`.
    - Tune the hyperparameters of each classifier using `GridSearchCV` with cross-validation.

    ```python
    from sklearn.datasets import load_breast_cancer

    # Load the breast cancer dataset
    breast_cancer = load_breast_cancer()
    ```

    ```python
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import GridSearchCV, train_test_split

    # Prepare data
    X, y = breast_cancer.data, breast_cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Decision Tree Classifier
    dtree = DecisionTreeClassifier()
    param_grid_dtree = {'max_depth': [3, 5, 7, 10]}
    grid_dtree = GridSearchCV(dtree, param_grid_dtree, scoring='f1', cv=5)
    grid_dtree.fit(X_train, y_train)
    y_pred_dtree = grid_dtree.predict(X_test)
    f1_dtree = f1_score(y_test, y_pred_dtree)
    auc_dtree = roc_auc_score(y_test, y_pred_dtree)
    print(f'Decision Tree F1: {f1_dtree}, AUC: {auc_dtree}')
    print(f'Best params for Decision Tree: {grid_dtree.best_params_}')

    # Random Forest Classifier
    rf = RandomForestClassifier()
    param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
    grid_rf = GridSearchCV(rf, param_grid_rf, scoring='f1', cv=5)
    grid_rf.fit(X_train, y_train)
    y_pred_rf = grid_rf.predict(X_test)
    f1_rf = f1_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_pred_rf)
    print(f'Random Forest F1: {f1_rf}, AUC: {auc_rf}')
    print(f'Best params for Random Forest: {grid_rf.best_params_}')

    # Gradient Boosting Classifier
    gb = GradientBoostingClassifier()
    param_grid_gb = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.5]}
    grid_gb = GridSearchCV(gb, param_grid_gb, scoring='f1', cv=5)
    grid_gb.fit(X_train, y_train)
    y_pred_gb = grid_gb.predict(X_test)
    f1_gb = f1_score(y_test, y_pred_gb)
    auc_gb = roc_auc_score(y_test, y_pred_gb)
    print(f'Gradient Boosting F1: {f1_gb}, AUC: {auc_gb}')
    print(f'Best params for Gradient Boosting: {grid_gb.best_params_}')
    ```

## Submission

- Submit the URL of the GitHub Repository that contains your work to NTU black board.
- Should you reference the work of your classmate(s) or online resources, give them credit by adding either the name of your classmate or URL.
