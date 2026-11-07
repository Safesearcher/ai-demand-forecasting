**AI Project: Deliverable 2 Guide** 

Model Architecture & Empirical Evaluation 

**Deadline: 10th November, 2026** 

Objective 

This deliverable shifts the focus from data foundations to the core ”intelligence” of your project.  You are required to implement your model architecture (Phase 3\) and conduct a rigorous  empirical evaluation (Phase 4). Furthermore, this milestone strictly evaluates your ability to  collaborate professionally using version control. 

**1 Phase 3: Model Architecture & Implementation** 

You must transition from data preprocessing to building the functional engine of your project. 

• **Framework Selection:** Clearly implement your solution using industry-standard frameworks  (*PyTorch, TensorFlow,* or *Scikit-Learn*). 

• **Design:** Define the layers, loss functions, and optimizers. 

• **Hyperparameters:** Document the tuning process. Provide a table of the parameters tested  (Learning rate, Batch size, Dropout, etc.) and justify your final selections. 

**2 Phase 4: Training & Evaluation** 

A model is only as good as its validated performance. 

• **Validation Strategy:** Implement a robust validation technique, such as K-Fold cross-validation or  a clear Train/Validation/Test split (e.g., 70/15/15). 

• **Metric Analysis:** Beyond simple accuracy, report Precision, Recall, F1-Score, and AUC-ROC. Justify  why these metrics are appropriate for your specific problem (especially if dealing with imbalanced  datasets). 

• **Error Analysis & Bias Check:** Identify where the model fails. Does it struggle with specific classes or demographic subgroups? Include a ”Bias Check” section in your report. 

**3 Professional GitHub & Collaboration Policy** 

As students of the 5th semester, you are expected to operate at an industry-standard level regarding  version control. Professionalism in Git is a core requirement of this deliverable. 

1 

Critical Git Requirements  
**No Manual Uploads:** It is strictly prohibited to use the ”Upload Files” button on the GitHub web  interface. You must use the Git Command Line (CLI) or a Desktop client to add, commit, push, and  sync changes. This is essential for effortless and professional collaboration. 

**Individual Coding Contribution:** Every group member is required to contribute code to the  project. I will evaluate individual participation by inspecting the **Git Commit History**. The logs  must clearly show which student implemented which specific module, function, or preprocessing  step. Each member must record their own work frequently; if a student’s coding contribution is  not visible in the Git history, they will receive a zero for the technical implementation phase. 

**4 Submission Checklist** 

Ensure the following are submitted: 

1\. **GitHub Repository:** Updated code in /src or /models folders. A requirements.txt file must be  present to ensure environment reproducibility. 

2\. **Model Weights:** Do not push large binary files (*\>*100MB) directly to Git. Provide a link in your  README.md to an external host (e.g., Google Drive or Hugging Face) where the weights can be  downloaded. 

3\. **LATEX Report:** An updated PDF covering Phase 3 and 4\. Include architecture diagrams,  loss/accuracy curves, and a **”Contribution Table”** that maps Git commit authors to specific  project modules. 

2