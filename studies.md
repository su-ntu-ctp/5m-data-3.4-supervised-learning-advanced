# **Self-Study Preparation Guide**

**⏳ Estimated Prep Time:** 45–60 minutes

Welcome to our flipped-classroom session, where you'll review foundational concepts beforehand to maximize our time for hands-on coding and debugging. This pre-study focuses on mastering **Regularization** and **Ensemble Methods**, ensuring you are ready to build robust, high-performance models that generalize well to unseen data.

## ⚡ Your Self-Study Tasks

Please complete the following activities before our session.

### 📝 Task 1: Controlling Complexity with Regularization (20 Minutes)

**Activity:** Read the **"Regularization in Machine Learning"** and **"Hyperparameter Tuning"** sections in the provided `supervised_learning_3_lesson.ipynb` notebook. Focus on understanding how we prevent models from memorizing noise (overfitting).

**Guiding Questions:**

* How does **L1 (Lasso)** regularization differ from **L2 (Ridge)** in terms of how they treat feature coefficients?
* Why is **Cross-Validation** considered a more reliable method for evaluating performance compared to a simple train-test split?

### 📝 Task 2: From Single Trees to Ensembles (25 Minutes)

**Activity:** Review the **"Decision Trees"** and **"Bagging vs Boosting"** sections in the notebook. Pay attention to how individual weak models are combined to create strong predictors.

**Focus your attention on these key components:**

1.  **Anatomy of a Tree:** How Gini Impurity or Entropy drives the splitting process.
2.  **Bagging (Random Forest):** How parallel training reduces variance.
3.  **Boosting (Gradient Boosting):** How sequential training reduces bias by correcting previous errors.

**Guiding Questions:**

* If you needed to reduce the risk of overfitting in a Decision Tree, what technique mentioned in the "Pruning" section would you apply?
* What is the fundamental structural difference between how a **Random Forest** is built versus **Gradient Boosting**?

### 📝 Task 3 (Optional): The Tuning Workflow

**Activity:** Briefly skim the **"Model Training Workflow"** and the **Grid Search** code implementation sections.

**Guiding Questions:**

* Why do we perform the train-test split *before* hyperparameter tuning?
* In the provided code, how does `GridSearchCV` help automate the process of finding the best `alpha` or `n_estimators`?

## 🙌🏻 Active Engagement Strategies

To deepen your retention, try one of the following while you review:

* **Concept Mapping:** Sketch a simple comparison table or diagram showing the differences between *Bagging* (Parallel) and *Boosting* (Sequential).
* **"Code Commentary":** Select the `GridSearchCV` code block in the Titanic example. Write a brief comment explaining what `cv=5` and `scoring='f1'` achieve in a real-world project.
* **Scenario Matching:** Imagine you have a dataset with 100 features, but you suspect only 5 are important. Based on the reading, would you use L1 (Lasso) or L2 (Ridge) regularization?

## 📖 Additional Reading Material

* [Parameter vs Hyperparameter](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/)
* [Scikit-Learn: Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)
* [Scikit-Learn: Ensemble methods (Random Forests & Gradient Boosting)](https://scikit-learn.org/stable/modules/ensemble.html)

### 🙋🏻‍♂️ See you in the session!

