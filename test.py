import numpy as np
import bentoml

ml_vl_runner = bentoml.sklearn.get("VL_Model:latest").to_runner()
ml_vl_runner.init_local()

data = np.array([0, 49, 3, 0, 5, 1])

data_2d = data.reshape(1, -1)

print(ml_vl_runner.predict.run(data_2d))