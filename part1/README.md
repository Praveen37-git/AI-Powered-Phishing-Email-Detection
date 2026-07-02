Downloaded the Phishing email dataset from Kaggle - https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector

The dataset contained no missing values, duplicate records, or incorrect data types. 
The corresponding functions were implemented to satisfy the assignment requirements and would correctly handle such cases if present. Since no missing values existed, no median imputation was applied.

All columns were correctly inferred as numeric (int64) by pandas. Therefore, no numeric-to-numeric conversion using pd.to_numeric() was required. Columns with a small set of repeated values (e.g., UsingIP, HTTPS, and class) were converted to the category dtype to reduce memory usage.

The dataset does not contain repetitive string columns. The target column (class) was converted to the categorical data type to reduce memory usage and reflect its categorical nature.

No outliers were removed because these values represent legitimate feature encodings rather than erroneous measurements, and removing them could discard useful information for the classifier.
