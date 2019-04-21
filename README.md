# PARIS

The traditional method to study application resilience to errors in HPC applications uses fault injection (FI), a time-consuming approach. While analytical models have been built to overcome the inefficiencies of FI, they lack accuracy. In this paper, we present PARIS, a machine-learning method to predict application resilience that avoids the time-consuming process of random FI and provides higher prediction accuracy than analytical models. PARIS captures the implicit relationship between application characteristics and application resilience, which is difficult to capture using most analytical models.

# Quantify Application Resilience by Doing Fault Injection

Please run FI campaigns to measure the application resilience. An FI campaign contains many FIs. In each FI, a single-bit error is injected into an input/output operand of an instruction. We classify the outcome, or manifestation, of programs corrupted by bit flips into three classes: success, SDC, and interruption:

* Success: The execution outcome is exactly the same as the outcome of the fault-free run. The execution outcome can also be different from the outcome of the fault-free run, but the execution successfully passes the result verification phase of the application.
* SDC: The program outcome is different from the outcome of the fault-free execution, and the execution does not pass the result verification phase of the application.
* Interruption (Exception): The execution does not reach the end of execution, i.e., it is interrupted in the middle of the execution, because of an exception, crash, or hang.
* Fault Manifestation Rates: To quantify the application resilience in an FI campaign, we measure the rate of each of the three classes of manifestations as the percentage of fault manifestations that are Success/SDC/Interruption within an FI campaign. The sum of three fault manifestation rates is 1. 

Please use PINFI (https://github.com/DependableSystemsLab/pinfi) to perform fault injections into programs because PINFI is accurate and user-friendly. To the end, please count fault manifestation rates for each program by examining program outcome. 

# Dynamic Instruction Trace Generation

Given an application, first generate its dynamic instruction trace using LLVM-Tracer (https://github.com/ysshao/LLVM-Tracer), a dynamic instruction trace generation tool.

# Feature Design and Extraction

For each application, please use its dynamic instruction trace as input to extract features. In this work, we use four instruction groups and resilience computation patterns as features. We also introduce execution order of dynamic instructions into features. It is also important to note that we consider resilience weight to differetiate resilience different between dynamic instruction instances and dyanmic pattern instances. See our paper for more details. 

* [1] Four instruction groups and four resilience computation patterns

For the four instruction groups (CFI, FPI, II, and MI) and the four patterns (Conditional statements, Shifting, Data Truncation, and Data Overwriting), please use the c++ code under ./Feature_construction/4_inst_type_4_resi_pattern_detection/4instype/ to count these four characteristics as features, using dynamic instruction trace as input.

* [2] The other two resilience computation patterns

For the other two patterns (Dead Corrupted Locations and Repeated Additions), please use the c++ code under ./Feature_construction/2_resi_pattern_detection/deadloc_detect/ and ./Feature_construction/2_resi_pattern_detection/repeatadd_detect/ to count Dead Corrupted Locations and Repeated Additions as features, using dynamic instruction trace as input.

* [3] Execution Order of Instructions

We use an example (See Figure 3 of the paper) to show that the execution order of instructions matters to error propagation and further to application resilience. To introduce order of instructions, we partition the dynamic instruction trace into chunks (each chunk is a gram). Each chunk is treated as a “word”, and the sequence of chunks is processed as the sequence of “words”. To the end, for each chunk, we collect four instruction groups and six resilience computation patterns to construct a foundation feature vector of 10 features. And we combine every two chunks to build a bigram feature vector of 20 features. In consequence, we build the final feature vector of size 30 by combining the foundation feature vector of size 10 and the bigram feature vector of size 20.

# Feature Construction and Implementation

* Given an application, the output of Step [1] is a file, in which there is an array of [N,M], where N is the number of chunks the dynamic instruction trace of the application is partitioned into; M is eight correlated to the four instruction groups and the four patterns.

* Given an application, the output of Step [2] has two files. 
- The file by running /deadloc_detect/ is the intermediate result with meta data. We then use ./Feature_construction/scripts_4_feature_vector_construction/extract_ratio.sh to extract the dead location rates (See Section 4.1.3 in the paper). To the end, a file with an array of [N,Q] is generated, where N is the number of chunks; Q is one because each chunk has a dead location rate. 
- The other file by running /repeatadd_detect/ has an array of [N,P], where P is one because we count number of repeat additions for each chunk.

* Combine the three arrays of [N,M], [N,Q], and [N,P], we get a big array of [N,M+Q+P]. We use ./Feature_construction/scripts_4_feature_vector_construction/combine_feature.sh to generate the big array.

* Taking order of instructions into account, according to Section 4.2 and Figure 4 in the paper, with the big array as input, we generate a feature vector of size 30 using the shell script ./Feature_construction/scripts_4_feature_vector_construction/feature_array_construction.py.

* The feature vector of size 30 is the final feature vector to the end. 

# Label Construction

* After running fault injection into an application, three values are generated: success rate, SDC rate, and interruption rate. 

* We predict three fault manifestation rates separately. 

- As an example, when predicting success rate of an application, using the extracted feature vector of size 30 as input to predict the success rate of the application.

# Dataset

* We construct a dataset of 116 applications: 100 small computation kernels and 16 big programs. 

* 100 small compuation kernels are manually created using resources from HackerRank (https://www.hackerrank.com/) and can be downloaded from https://github.com/HPCCS/Computation-Kernel-Dataset. 

* The 16 big programs are from representative benchmark suites and proxy applications (See Table 2 in the paper). The big programs can easily be downloaded from the Internet.  

* We use the 100 small computation kernels for training and the 16 big programs for testing. 

# Training and Testing

* Using the python code ./Machine_learning_model/GBR_model.py for training and predicting. 
- The input is the feature array of the training data, the label array of the training data, the feature array of the testing data, and the label array of the testing data. 

- The feature array of the training data is [X,30]; X is the number of programs within the training dataset. X is 100 in our case. The label array of the training data is [Y,1]; Y is the number of programs within the training dataset. The logic is the same for the feature array of the testing data and the label array of the testing data.

- The output is the predicted fault manifestation rates. 

* We run the prediction model for two times to predict success rate and interruption rate, and we count SDC rate by subtracting success rate and interruption from 1. We do not directly predict SDC rate because values of SDC rates are zero for many small computation kernels which significantly affects the training accuracy.

* We use Mean Absolute Percentage Error (MAPE) to count prediction error because MAPE is commonly used for regression model evaluations and can intuitively interpret modeling accuracy in terms of relative errors.

# Other code and scripts

* Under ./Machine_learning_model/, there are other code files: feature_selection.py, top_k_expt.py, top_k_succ.py, and model_select.py. The first three are used for feature sorting and feature selection; the last one is used for model selection. 

* Under ./Feature_construction/, there are other three script files: 4instype.sh, deadloc.sh, and repeatadd.sh. They help to process many dynamic trace files in batch. 

