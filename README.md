# Machine_Learning-Pipeline
Creating a Machine Learning pipeline for evaluating a tabular dataset, including data normalisation, feature/instance selection, class imbalance correction, several (appropriate) machine learning models, hyperparameter tuning and cross-validation evaluation.

The code follows a comprehensive machine learning pipeline for evaluating various algorithms and approaches on an imbalanced dataset. Here’s a summary of what each part achieves and some observations about the results:

### 1. Decision Tree Model and Optimization

- **Base Decision Tree**: The initial model achieves an accuracy of about 72.77% on oversampled data.
- **Optimized Decision Tree**: After tuning hyperparameters, accuracy slightly improves to 72.83%, indicating a modest benefit from optimization.
  
### 2. Random Forest Model and Optimization

- **Base Random Forest**: Achieves a better accuracy (74.8%) compared to the Decision Tree, highlighting that Random Forest tends to perform better due to ensemble learning.
- **Optimized Random Forest**: Tuning results in a slight improvement to 74.85%, showcasing the model’s robustness but also that it may have diminishing returns with further tuning.

### 3. Support Vector Machine (SVM)

- **Performance**: The SVM model performs poorly, achieving only 44% accuracy and struggling with the multiclass problem due to imbalance and likely feature scaling issues.
  
### 4. Normalization

- **Scaling**: Data normalization (MinMaxScaler) scales features to a range [0,1], which is helpful for algorithms sensitive to feature scale (like SVMs). However, further tuning and alternative models might be necessary for SVM to perform better.

### 5. Decision Tree on Non-Oversampled Data

- **Performance Comparison**: Using non-oversampled data, the Decision Tree model achieves 73.85%, indicating that oversampling helped improve representation and possibly model performance in this context.

### 6. Cross-Validation with Decision Tree

- **Purpose**: Cross-validation checks the model’s robustness across different subsets of the data, providing more insight into its generalizability.

### Future Work Suggestions:

1. **Experiment with Other Models**: Trying algorithms like Gradient Boosting or XGBoost may yield better results, especially for imbalanced datasets.
2. **Consider Alternative Sampling**: Try other sampling techniques like SMOTE or ADASYN to see if they improve the minority class representation and overall accuracy.
3. **Further Hyperparameter Tuning**: For SVM, consider kernel tuning or regularization adjustments to see if this improves performance.

This workflow is well-rounded and could be further enhanced by exploring ensemble techniques or advanced imbalanced learning strategies.
