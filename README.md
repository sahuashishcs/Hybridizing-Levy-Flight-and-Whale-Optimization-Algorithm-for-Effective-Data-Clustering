# README.txt

Title: Hybridizing Levy Flight and Whale Optimization Algorithm for Effective Data Clustering

Publication: Arabian Journal for Science and Engineering

------------------------------------------------------------

## 1. Description
- This MATLAB code is associated with the manuscript entitled “Hybridizing Levy Flight and Whale Optimization Algorithm for Effective Data Clustering”.
- The base implementation corresponds to the Whale Optimization Algorithm (WOA) described in Reference 17 of the manuscript.
- The code has been modified by **Dr. Ashish Kumar Sahu** and **Dr. Tribhuvan Singh** to solve data clustering problems.

------------------------------------------------------------

## 2. Requirements
- MATLAB (latest version recommended).
- Standard MATLAB toolboxes (no additional packages required).

------------------------------------------------------------

## 3. Usage Instructions
1. Download and extract the code files.
2. Open MATLAB and set the working directory to the code folder.
3. To run the algorithm, execute the script `LWOA.m`.
4. In order to change the dataset:
   - Load the dataset name in the code.
   - Modify the dataset dimensions (`dim`) and number of clusters (`k`).
   - For datasets where the actual cluster ID for each data point is provided in the last column, use the command:
     ```matlab
     X = X(:,1:dim);
     ```
5. The program will output clustering results along with performance metrics.

------------------------------------------------------------

## 4. Reference
If you use this code in your research, please cite the following paper:
- Singh, T., Sahu, A.K. *Hybridizing Levy Flight and Whale Optimization Algorithm for Effective Data Clustering*. Arabian Journal for Science and Engineering.

------------------------------------------------------------

## 5. Contact
For queries or clarifications, please contact:
  **Dr. Ashish Kumar Sahu**
- **Dr. Tribhuvan Singh**
- Email: [Given in the manuscript]
