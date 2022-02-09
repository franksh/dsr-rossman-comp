import os
import pickle
from sklearn.pipeline import Pipeline

pipelines_directory = '../data/trained_pipelines/'


def load_pipeline(name: str) -> Pipeline:
    """ Load a pipeline by its name.
    
    Parameters:
    -----------
    - name: str
        The name of the pipeline. Has to correspond to
        one of the file name of pipelines stored in /pipelines

    Returns:
    --------
    - pipeline: sklearn.pipeline.Pipeline
        The (trained) sklearn pipeline object.
    """

    # Set up paths
    filename = f'pipeline_{name}.p'
    file_path = pipelines_directory + filename

    # Load the model
    print(f' - Loading pipeline "{name}" at:\n{file_path}')
    try:
        loaded_pipeline = pickle.load(open(file_path, 'rb'))
    except FileNotFoundError as e:
        print(f'Error: Could not find pipeline "{name}" at path:\n{file_path}')
        return

    return loaded_pipeline

def save_pipeline(pipeline: Pipeline, name: str):
    """ Save a (trained) pipeline.

    The pipeline object will be pickled under the given name.

    Parameters:
    -----------
    - pipeline: object, instance of any sklearn pipeline class
        The (trained) sklearn pipeline object.
    - name: str
        The the name under which the pipeline will be stored
        and can be retreived.
    """
    # Create pipeline save path if it doesn't exists
    if not os.path.exists(pipelines_directory):
        os.makedirs(pipelines_directory)

    # Set up paths
    filename = f'pipeline_{name}.p'
    file_path = pipelines_directory + filename
    
    # Save pipeline
    print(f' - Saving pipeline "{name}" at:\n{file_path}')
    pickle.dump(pipeline, open(file_path, 'wb'))


