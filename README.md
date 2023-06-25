# ml-pipeline-tf
intern assignment
This code provides the functionality to deploy machine learning models using a RESTful API. To use this code, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies by running pip install -r requirements.txt.
3. Start the FastAPI server by running the command uvicorn app:app --reload in your terminal.
4. The /status endpoint can be accessed by navigating to http://localhost:8000/status in your web browser.
5. To make predictions using the deployed models, send a POST request to the /predict endpoint with the following JSON payload:


   {
       "hf_pipeline": "<pipeline_type>",
       "model_deployed_url": "<url_of_deployed_model>",
       "inputs": "<input_data>",
       "parameters": {
           "parameter1": "<value1>",
           "parameter2": "<value2>"
       }
    }

  where:

    •hf_pipeline: The type of pipeline to use for prediction (text-generation, zero-shot-classification, token-classification, or object-detection).

    •model_deployed_url: The URL of the deployed model to be used for prediction.

    •inputs: The input data to be used for prediction (e.g. text, image URL, etc.).

    •parameters: Any additional parameters required by the model.

  7.The response will be a JSON object containing the predicted output based on the input data.
