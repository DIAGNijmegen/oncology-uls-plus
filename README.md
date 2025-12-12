
Dataset can be found on [Hugging Face](https://huggingface.co/datasets/nielsRocholl/ULS_plus/) or [Zotero]()

Each dataset in `archives/` has been preprocessed with our [preprocessing pipeline](preprocessing/README.md) located in this repository.

Once the data is downloaded you can follow the [training instrcuctions](training/train.md) to train the model with nnUNetv2. 

Once trained, or using our [provided model weights](https://huggingface.co/nielsRocholl/ULS_plus/tree/main) you can run our [evaluation script](evaluation/README.md)