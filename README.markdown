# NN-Classifier-CIFAR

## Project Overview
This project implements and compares two neural network classifiers for image classification on dandelions and grass plants using the CIFAR dataset:
- **Fully-Connected Neural Network (FCNN)**: A dense neural network for baseline classification.
- **Convolutional Neural Network (CNN)**: A convolutional model leveraging spatial features for improved performance.

Built with TensorFlow/Keras, the project evaluates model performance using accuracy, ROC-AUC, and complexity, and includes visualizations like ROC curves. Additional metrics like Precision-Recall curves and F1-Score are suggested for deeper performance analysis, especially for imbalanced datasets.

## Features
- **Dataset**: CIFAR dataset (dandelions and grass plants, 32x32 RGB images).
- **Models**:
  - FCNN: Dense layers for image classification.
  - CNN: Conv2D, MaxPooling2D, and Dense layers for enhanced feature extraction.
- **Evaluation Metrics**:
  - Accuracy
  - ROC-AUC (FCNN: 0.73, CNN: 0.75)
  - Suggested: Precision-Recall Curve, F1-Score, Confusion Matrix
- **Visualization**: ROC curves comparing FCNN and CNN performance.
- **Tools**: MLflow and Weights & Biases (wandb) for experiment tracking.

## Repository Structure
- `Building_a_Fully_Connected_and_Convolutional_NN_Classifier.ipynb`: Jupyter Notebook with data preprocessing, model implementation, training, evaluation, and ROC curve visualization.
- `README.md`: This file, providing project overview and instructions.
- `LICENSE`: MIT License file for open-source usage.

## Requirements
To run the notebook, you need the following dependencies:
- Python 3.6+
- TensorFlow/Keras
- NumPy
- Matplotlib
- MLflow
- Weights & Biases (wandb)

Install dependencies using:
```bash
pip install tensorflow numpy matplotlib mlflow wandb
```

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/NN-Classifier-CIFAR.git
   cd NN-Classifier-CIFAR
   ```

2. **Set Up Environment**:
   - Ensure dependencies are installed.
   - (Optional) Configure MLflow and wandb accounts for experiment tracking.

3. **Run the Notebook**:
   - Open `Building_a_Fully_Connected_and_Convolutional_NN_Classifier.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute cells sequentially to:
     - Install dependencies.
     - Load and preprocess the CIFAR dataset.
     - Build, train, and evaluate FCNN and CNN models.
     - Visualize ROC curves and compare performance.

4. **Evaluate Results**:
   - Check accuracy and ROC-AUC scores (CNN slightly outperforms FCNN with 0.75 vs. 0.73).
   - Consider Precision-Recall curves for imbalanced data insights.

## Results
- **Accuracy**: Both models achieve comparable accuracy, with CNN slightly better due to spatial feature extraction.
- **ROC-AUC**:
  - FCNN: 0.73
  - CNN: 0.75 (preferred for reliability across thresholds)
- **Complexity**: CNN is more complex but justified by marginal performance gains unless simplicity is prioritized.
- **Recommendations**:
  - Use Precision-Recall curves for imbalanced datasets.
  - AUC-PR and F1-Score provide robust metrics for positive class prediction.

## Future Improvements
- Experiment with deeper CNN architectures or hyperparameter tuning.
- Implement Precision-Recall curves and Confusion Matrix for detailed evaluation.
- Explore data augmentation to improve model robustness.
- Optimize model complexity for resource-constrained environments.

## Contributing
Contributions are welcome! To contribute:
- Open issues for bugs or feature suggestions.
- Submit pull requests with improvements, ensuring code aligns with the MIT License.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- CIFAR dataset for providing the image data.
- TensorFlow/Keras for deep learning framework.
- MLflow and Weights & Biases for experiment tracking.