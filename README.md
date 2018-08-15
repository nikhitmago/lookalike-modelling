# lookalike-modelling

Versions: Spark: 2.1.1, Python: 3.5

- Objective of the project is to find customer lookalikes (similar customers) in a large population using algorithms like Random Forest, KNN and KD-Trees
- Cannot share data as it is sensitive to clients but need to follow the following format:

Train Data:
col 1 -> ID
col 2...n-1 -> features
col n -> Label (Binary)

Test Data:
col 1 -> ID
col 2 -> features

- The train code will treat 1 as minority class and generate the 30% customer IDs that are lookalikes after running predict.py
- Please manually fill the file locations in the beginning the code for input files, intermediate files etc.
