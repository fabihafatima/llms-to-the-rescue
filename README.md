# LLMs to the Rescue: Can Text Style Transfer bridge the Gap in Minority Class Detection?

## Project Overview
This project tackles the challenge of class imbalance in text classification tasks, particularly when certain classes are underrepresented in the dataset. By leveraging Large Language Models (LLMs) for data augmentation through style transfer techniques, we aim to enhance class balance and improve model performance.

We utilize the AG News dataset and introduce a novel approach to generate synthetically diverse text samples while preserving contextual relevance. This involves two distinct augmentation strategies and evaluation using RoBERTa and Logistic Regression classifiers.

## Key Features
- **Dataset:** AG News with an intentionally imbalanced class distribution.
- **Augmentation Techniques:**
  1. **Baseline Model:** Generates simple text variations.
  2. **Advanced Model:** Applies style transfer to create variations across journalistic styles (e.g., Investigative, Editorial, Breaking News).
- **Evaluation Models:**
  - RoBERTa
  - Logistic Regression
- **Objective:** Improve classification accuracy, especially for minority classes, through high-quality, diverse augmented samples.

## Results
The results demonstrate that style transfer via LLMs for data augmentation:
- Significantly improves classification accuracy across all classes.
- Provides an effective and scalable solution to address class imbalance in NLP tasks.

## Project Structure
The following steps are organized into separate cells within the same file, `llms_to_the_rescue.ipynb`:
1. **`data_preprocessing`:** Contains data preparation steps, including handling the imbalanced dataset.
2. **`augmentation_baseline`:** Implements the baseline augmentation strategy.
3. **`augmentation_style_transfer`:** Implements the advanced style transfer-based augmentation.
4. **`model_training`:** Trains and evaluates the classifiers.
5. **`results_analysis`:** Analyzes and visualizes the results of the augmentation methods.

## Getting Started
1. **Environment Setup:**
   - Open the `.ipynb` files in Google Colab.
   - Install the required Python packages by running the following command in a Colab cell:
     ```bash
     !pip install transformers sklearn matplotlib pandas
     ```

2. **Dataset Preparation:**
   - The AG News dataset is used for this project. Download it from [Kaggle](https://www.kaggle.com/amananandrai/ag-news-classification-dataset) or another source and upload it to your Colab environment.

3. **Run Notebooks:**
   - Follow the sequence listed in the project structure to preprocess data, augment it, train models, and evaluate results.

## Usage
- **Data Augmentation:** Use the `augmentation_baseline` and `augmentation_style_transfer` cells to generate augmented data for other NLP tasks.
- **Classification Models:** Adapt the `model_training` cell for custom text classification tasks.

## Dependencies
- Python 3.x
- Google Colab
- Required Libraries: `transformers`, `sklearn`, `matplotlib`, `pandas`

## Limitations
- This project focuses on the AG News dataset; results may vary with other datasets.
- The effectiveness of augmentation depends on the quality of prompts used for LLMs.

## Future Work
- Extend the style transfer approach to more journalistic styles.
- Explore additional LLMs and augmentation techniques.
- Evaluate the methodology on other NLP datasets and tasks.

## Acknowledgments
- AG News dataset: [Kaggle](https://www.kaggle.com/amananandrai/ag-news-classification-dataset)
- Hugging Face Transformers library: [Hugging Face](https://huggingface.co/transformers/)

---
For any questions or suggestions, feel free to contact us!

