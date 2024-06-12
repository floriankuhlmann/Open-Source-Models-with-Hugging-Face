
# Hugging Face Models Experimentation

This repository contains a Python project based on the [DeepLearning.AI course](https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/). The project utilizes Jupyter Notebooks to implement and experiment with various open-source models from Hugging Face.

## Project Overview

This learning project aims to apply the knowledge gained from the course to create a Python application that leverages Hugging Face models for various tasks. It is designed to experiment with and understand the capabilities of the Hugging Face transformers library.

### What You'll Learn

- **NLP Tasks**: Create chatbots, translate languages, summarize documents, and measure text similarity.
- **Audio Tasks**: Convert audio to text (ASR), text to audio (TTS), and perform zero-shot audio classification.
- **Image Tasks**: Generate audio narrations for images, perform zero-shot image segmentation, and implement visual question answering, image search, and image captioning.
- **Multimodal Tasks**: Combine models to perform tasks that involve multiple types of data inputs and outputs.

### Tools Used

- **Transformers Library**: Utilize pre-trained models from Hugging Face for various tasks.
- **Gradio**: Create user-friendly interfaces for your applications.
- **Hugging Face Spaces**: Deploy your applications on the cloud for easy sharing and accessibility.

### Course Details

The course provides building blocks to combine into a pipeline for AI-enabled applications. You will learn to:

- Use transformers for NLP tasks.
- Convert between audio and text.
- Perform image and multimodal tasks.
- Share your applications using Gradio and Hugging Face Spaces.

### Who Should Join?

Anyone interested in quickly and easily building AI applications using open-source models.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/huggingface-experiments.git
   ```
2. Navigate to the project directory:
   ```bash
   cd huggingface-experiments
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   for the use of the models gardiospeech_model.py and librispeech_model.py you need to install ffmpeg from https://ffmpeg.org
   download the windows installer from the web or on mac os x do:
   ```bash
   brew install ffmpeg
   ```
4. Explore the app to understand and run the experiments.

## Contributions

Feel free to fork this repository, create issues, or submit pull requests to improve the project.

## License

This project is licensed under the MIT License.

---

Happy experimenting with Hugging Face models!
