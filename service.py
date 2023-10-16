import numpy as np
import bentoml
from bentoml.io import NumpyNdarray



ml_vl_runner = bentoml.sklearn.get("VL_Model:latest").to_runner()

#creating a service with name VL_Classifier
decision_tree = bentoml.Service("VL_Classifier",runners=[ml_vl_runner])

#The service.py code is annotated with the service access point (input/output)
@decision_tree.api(input=NumpyNdarray(),output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = ml_vl_runner.predict.run(input_series)
    if result == 1:
        print("Suppressed")
    else:
        print("Unsuppressed")
    
    return result


#bentoml serve service.py:decision_tree --reload
#bentoml serve service.py:decision_tree --port 8080