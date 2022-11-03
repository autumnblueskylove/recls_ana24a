# SIA **Cl**a**s**sification Framework

## Introduction

**recls** is a SI-Analytics's classification framework based on [mmclassification](https://github.com/open-mmlab/mmclassification).

Main goal of this project is to provide a maintanable, well-moduled, reliable classification framework **especially focusing on satellite and aerial imagery domain**.

If you are new to **mm**-frameworks, we recommend you to read documentations in mmclassification as below;

- [mmclassification/Getting Started](https://mmclassification.readthedocs.io/en/latest/getting_started.html)

<details>
<summary>Major Features</summary>

- **MLOps Pipelines**
  We provide mlops-related below features:

  1. MLFlowLogger to track experiment and save checkpoint
  2. Batch Inference with MLFlow Run-ID
  3. Evaluation with MLFlow Run-ID

- **Domain-related Features**

  1. Batch inference for large satellite imagery with efficient memory cost and inference time

- **Dataset**

  1. MonogoDB
  2. DatasetPlatform
     - Install DpClient
       - `pip3 install --index-url https://pypi.sia-service.kr/simple/ --upgrade dp-client `

</details>

# Common scripts

You can check most common scripts and tools in the [tools/README.md](tools/README.md).
