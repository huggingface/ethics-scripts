## `bart-large-cnn-samsum`

> If you want to use the model you should try a newer fine-tuned FLAN-T5 version [philschmid/flan-t5-base-samsum](https://huggingface.co/philschmid/flan-t5-base-samsum) out socring the BART version with `+6` on `ROGUE1` achieving `47.24`.

# TRY [philschmid/flan-t5-base-samsum](https://huggingface.co/philschmid/flan-t5-base-samsum)


This model was trained using Amazon SageMaker and the new Hugging Face Deep Learning container.

For more information look at:
- [🤗 Transformers Documentation: Amazon SageMaker](https://huggingface.co/transformers/sagemaker.html)
- [Example Notebooks](https://github.com/huggingface/notebooks/tree/master/sagemaker)
- [Amazon SageMaker documentation for Hugging Face](https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html)
- [Python SDK SageMaker documentation for Hugging Face](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/index.html)
- [Deep Learning Container](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-training-containers)

## Hyperparameters
```json
{
    "dataset_name": "samsum",
    "do_eval": true,
    "do_predict": true,
    "do_train": true,
    "fp16": true,
    "learning_rate": 5e-05,
    "model_name_or_path": "facebook/bart-large-cnn",
    "num_train_epochs": 3,
    "output_dir": "/opt/ml/model",
    "per_device_eval_batch_size": 4,
    "per_device_train_batch_size": 4,
    "predict_with_generate": true,
    "seed": 7
}
```

## Usage
```python
from transformers import pipeline
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

conversation = '''Jeff: Can I train a 🤗 Transformers model on Amazon SageMaker? 
Philipp: Sure you can use the new Hugging Face Deep Learning Container. 
Jeff: ok.
Jeff: and how can I get started? 
Jeff: where can I find documentation? 
Philipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face                                           
'''
summarizer(conversation)
```

## Results

| key | value |
| --- | ----- |
| eval_rouge1 | 42.621 |
| eval_rouge2 | 21.9825 |
| eval_rougeL | 33.034 |
| eval_rougeLsum | 39.6783 |
| test_rouge1 | 41.3174 |
| test_rouge2 | 20.8716 |
| test_rougeL | 32.1337 |
| test_rougeLsum | 38.4149 |