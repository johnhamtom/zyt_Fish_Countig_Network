# LDNet: High Accuracy Fish Counting Framework using Limited training samples with Density map generation Network

![](README_md_files/b2eb5560-507f-11ee-ab89-8d5e8495a351.jpeg?v=1&type=image)

## Related instructions

&#x20; **./carDatasets** stores 5%-10% of the selected CARPK and PURKP+ training data sets and related processing codes. **./show** stores high-density fry pictures and density maps. For the project structure, please refer to [GL ](https://github.com/jia-wan/GeneralizedLoss-Counting-Pytorch).

&#x20; First run **IMBDA.py** to perform data enhancement on the training data, then run **PDdata.py** to prepare the training data, and finally run train.py for training.

&#x20; We provide training weights for the vehicle dataset, and you can run **testCar.py** to test with the weights. The test set is consistent with the open source data sets CARPK and PURKP+.
