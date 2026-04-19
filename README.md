<div align="center"><h1> Diabetes Diagnosis: A Machine Learning Approach </h1>


</div>

_Diabetes, resulting from inadequate insulin production or
utilization, causes extensive harm to the body. Existing diagnostic methods are often invasive and come with drawbacks, such as cost constraints.
Although there are machine learning models like Classwise k Nearest
Neighbor (CkNN) and General Regression Neural Network (GRNN),
they struggle with imbalanced data and result in underperformance.
Leveraging advancements in sensor technology and machine learning,
we propose a non-invasive diabetes diagnosis using a Back Propagation
Neural Network (BPNN) with batch normalization and XGBoost for feature normalization, incorporating data
re-sampling and normalization for class balancing. Our method addresses
existing challenges such as limited performance associated with traditional machine learning. Experimental results on three datasets show
significant improvements in overall accuracy, sensitivity, and specificity
compared to traditional methods. Notably, we achieve accuracies of **90%**
in Pima diabetes dataset. This underscores the potential of advanced deep learning models, including Transformers, for robust diabetes
diagnosis._

![main](static/images/diabetes.svg)

## Code

See [main_new.ipynb]. To get started, you must open up the file and install everything in requirements.txt. Do not forget to install scikeras, otherwise the code will not work. From there, run all of the blocks and watch the magin unravel.

## Citation

```
@inproceedings{zhang2024deep,
  title={A deep learning approach to diabetes diagnosis},
  author={Zhang, Zeyu and Ahmed, Khandaker Asif and Hasan, Md Rakibul and Gedeon, Tom and Hossain, Md Zakir},
  booktitle={Asian Conference on Intelligent Information and Database Systems},
  pages={87--99},
  year={2024},
  organization={Springer}
}
```
