# Get the basic image
FROM continuumio/miniconda3

# Directory with all docker image related files
WORKDIR /docker_image

# Create the environment
COPY environment.yml .
RUN conda env create -f environment.yml

# To activate the environment(this runs inside the environment created)
SHELL ["conda", "run", "-n", "corner_shop_ml_env", "/bin/bash", "-c"]
RUN echo "Environment is activated, dependencies are installed."

# The code to run when container is started
COPY main.py .
COPY __init__.py .
COPY config .
COPY Google_SA.json .
COPY preprocessing_layer ./preprocessing_layer
COPY modelling_layer ./modelling_layer

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "corner_shop_ml_env", \
            "python", "main.py", "--run_local", "False", "--bigqueryData", "True"]
