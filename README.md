# Image-Analysis-

Hybrid Model Approach: CNN-SVM Fusion for CIFAR-10 Classification
Introduction to Project
This project focuses on implementing a hybrid approach for image classification using the CIFAR-10 dataset. The methodology involves the integration of a Convolutional Neural Network (CNN) and a Support Vector Machine (SVM) to achieve accurate and efficient classification.

Data Preprocessing and Dataset Splitting
The CIFAR-10 dataset is loaded using Keras, and preprocessing steps include normalization of pixel values and one-hot encoding of target labels. The dataset is then split into training, validation, and test sets to facilitate model training, feature extraction, and final evaluation.

CNN Model Architecture and Training Details
A CNN model is constructed using Keras, featuring two convolutional layers, two max-pooling layers, and dropout layers to prevent overfitting. The rectified linear unit (ReLU) activation function is applied in convolutional layers, and the softmax activation function is used in the output layer. The model is compiled with categorical cross-entropy loss and the Adam optimizer.

Feature Extraction from CNN Model
To enhance the SVM model's efficiency, features are extracted from the fully connected layer before the output layer using the Keras functional API. This reduces data dimensionality and improves SVM training time.

SVM Model Training and Evaluation
An SVM model with a linear kernel is trained on the extracted features from the validation set. The trained model predicts classes for the test set, and its performance is evaluated using accuracy, F1 score, and recall.

Performance Metrics: Accuracy, F1 Score, and Recall
The performance of both the CNN and SVM models is assessed using common metrics such as accuracy, F1 score, and recall. Similarities in performance metrics between the two models indicate the efficacy of feature extraction.

Comparison between CNN and SVM Models
A comprehensive comparison is presented, highlighting similarities and differences in accuracy, F1 score, and recall between the CNN and SVM models. This analysis provides insights into the effectiveness of the hybrid approach.

Conclusion and Insights
The project concludes with a summary of findings, emphasizing the comparable performance of the CNN and SVM models. Insights are drawn regarding the benefits of feature extraction for reducing dimensionality and improving training efficiency.

Challenges and Alternative Experiments
The README discusses challenges faced during the project, such as time-consuming SVM model compilation and complexities encountered with the CIFAR-100 dataset. Alternative experiments using the CIFAR-10 dataset are justified.

Future Considerations and Applications
Suggestions for future work and potential applications of the hybrid model approach are outlined. The README encourages further exploration and refinement of the methodology.

Acknowledgments and References
Acknowledgments for datasets, libraries, and other resources used in the project are provided. References to relevant literature or code snippets are also included.
