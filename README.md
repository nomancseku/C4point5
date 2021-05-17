# C4.5 Algorithm implementation. 
## This work is a part of my Pattern Recognition Lab on CSE, KHULNA UNIVERSITY.
## Abdullah Al Noman, CSE-KU

### Hello there,
### To use this module, open command prompt and type - *```pip install C4point5```*

after installing the module write code as below,
```
from C4point5 import C45

model = C45.C4point5([train dataset path])

model.readDataset()
model.preprocessData()
model.generateTree()
model.printTree()
model.predict()
model.evaluate([test dataset path])
model.viewTruePred([test dataset path])
```
