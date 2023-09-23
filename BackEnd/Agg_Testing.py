

# sci-kit learn libraries
import sklearn
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import hinge_loss

# dataset and ml libraries
from keras.datasets import cifar10
import torch
import tensorflow
import onnx

import numpy as np

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# miscellaneous
import time
import random
import sys
from pdb import set_trace

# Set seeds for poison / camou / target selection here:
random.seed(222222)

# Class Dictionary for CIFAR10
classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
             'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

binaryClasses = {0:'Machine', 1:'Animal'} # Machine , Animal

# Method to obtain camouflages: use "BREW" for Gradient matching or "FLIP" by label flipping
CAMO = 'BREW'

(x_train, y_train_label), (x_test, y_test_label) = cifar10.load_data()

def rgb_to_gray(images):
    return np.dot(images[...,:3], [0.2989, 0.5870, 0.1140])

# Convert RGB to grayscale
x_train = rgb_to_gray(x_train)
x_test = rgb_to_gray(x_test)

# Expand dimensions for grayscale channel (making it 32x32x1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Normalize pixel values and dividing each image by its Frobenius norm
x_train = x_train / 255.0
x_test = x_test / 255.0
m_train, m_test = x_train.shape[0], x_test.shape[0]

x_train_flat = x_train.reshape(m_train, -1)
x_test_flat = x_test.reshape(m_test, -1)
x_train_norm = np.linalg.norm(x_train_flat, ord=2, axis=1, keepdims=True)
x_test_norm = np.linalg.norm(x_test_flat, ord=2, axis=1, keepdims=True)

x_train_flat = x_train_flat / x_train_norm
x_test_flat = x_test_flat / x_test_norm

# Flatten labels
y_train_label = y_train_label.ravel()
y_test_label = y_test_label.ravel()
animal_indices = []
machine_indices = []

y_train = np.array([False] * m_train)
y_test = np.array([False] * m_test)

# Convert to Binary dataset
for i in range(m_test):
    if y_test_label[i] in [2,3,4,5,6,7]:
        y_test[i] = True
    else:
        y_test[i] = False

for i in range(m_train):
    if y_train_label[i] in [2,3,4,5,6,7]:
        y_train[i] = True
        animal_indices.append(i)
    else:
        y_train[i] = False
        machine_indices.append(i)

start_time = time.time()

#model_SVC = SVC(kernel = 'linear', max_iter=100, probability=True)
clean_model = LinearSVC(loss='hinge', max_iter=3000)
#fit
clean_model.fit(x_train_flat, y_train)

print("Elapsed[s] : ", time.time() - start_time)

def show_image(img, norm = None):
    if norm:
        #To show Image:
        target = img * norm
        plt.imshow(target.reshape(32, 32, 1))
        plt.show()
    else:
        plt.imshow(img.reshape(32, 32, 1))
        plt.show()

close_pos = []

# Run model on Test set

decision_function_test = clean_model.decision_function(x_test_flat)

# Find all support vectors of the model
close_positives = np.where((decision_function_test <= 0.1) & (decision_function_test > 0))[0]
print(close_positives)

# Making sure the target is chosen from animals but prediction is machine
# score > 0: Machine as Class 0
# score < 0: Animal  as Class 1
for ind in close_positives:
  if y_test[ind] == 1:
    close_pos.append(ind)

target_indice = random.choices(close_pos)

print('target_indice:', target_indice)
target_label = 0
target = x_test_flat[target_indice[0]].reshape(1, -1)
target_original_label = y_test[target_indice[0]]

#To show Target Image:
plt.imshow(target.reshape(32, 32, 1) * x_test_norm[target_indice[0]])
print("Target chosen from class:", binaryClasses[target_original_label])
print("Target assigned to class:", binaryClasses[target_label])

# Deleting Target from orignal test set.
x_test = np.delete(x_test_flat, target_indice[0], 0)
target_original_label = y_test[target_indice[0]]
y_test = np.delete(y_test, target_indice[0], 0)

# check if notebook is in colab
try:
    # install ezkl
    import google.colab
    import subprocess
    #import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ezkl"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sk2torch"])

# rely on local installation of ezkl if the notebook is not in colab
except:
    pass


# here we create and (potentially train a model)

# make sure you have the dependencies required here already installed
import json
#import numpy as np
from sklearn.svm import SVC
import sk2torch
#import torch
import ezkl
import os

import warnings
warnings.filterwarnings('ignore')

# check if notebook is in colab
try:
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "solc-select"])
    # 执行其他 shell 命令
    subprocess.run(["solc-select", "install", "0.8.20"], check=True)
    subprocess.run(["solc-select", "use", "0.8.20"], check=True)
    subprocess.run(["solc", "--version"], check=True)

# rely on local installation if the notebook is not in colab
except:
    pass

model_path = os.path.join('network.onnx')
compiled_model_path = os.path.join('network.compiled')
pk_path = os.path.join('test.pk')
vk_path = os.path.join('test.vk')
settings_path = os.path.join('settings.json')
srs_path = os.path.join('kzg.srs')
witness_path = os.path.join('witness.json')
data_path = os.path.join('input.json')
proof_path = os.path.join('proof.json')

aggregate_proof_path = os.path.join('aggr.pf')
aggregate_vk_path = os.path.join('aggr.vk')
aggregate_pk_path = os.path.join('aggr.pk')

async def async_function(data_path, model_path, settings_path, resource_string):
    res = await ezkl.calibrate_settings(data_path, model_path, settings_path, resource_string)
    assert res == True

def gen_verifier(wrap_model, input_image, expected_output):
    print ('Challenge Image:')
    show_image(input_image, x_test_norm[target_indice[0]])

    model = sk2torch.wrap(wrap_model)
    x = torch.from_numpy(input_image)
    torch_out = model.predict(x)

    val = torch_out.item()

    if val != expected_output:
        #print (torch_out)
        print ('Error! Model trained by dataset predicting this image as animal is', val)
        return

    print ('Basic check passes, generate ZKML verifier')
    torch.onnx.export(model,               # model being run
                    # model input (or a tuple for multiple inputs)
                    x,
                    # where to save the model (can be a file or file-like object)
                    "network.onnx",
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['input'],   # the model's input names
                    output_names=['output'],  # the model's output names
                    dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                    'output': {0: 'batch_size'}})

    d = ((x).detach().numpy()).reshape([-1]).tolist()

    data = dict(input_shapes=[target.shape[1:]],
                input_data=[d],
                output_data=[o.reshape([-1]).tolist() for o in torch_out])

    # Serialize data into file:
    json.dump(data, open("input.json", 'w'))


    # TODO: Dictionary outputs
    res = ezkl.gen_settings(model_path, settings_path)
    assert res == True

    res = async_function(data_path, model_path, settings_path, "resource")
    #res = await ezkl.calibrate_settings(data_path, model_path, settings_path, resource_string)
    #assert res == True

    res = ezkl.compile_model(model_path, compiled_model_path, settings_path)
    assert res == True

    # srs path
    res = ezkl.get_srs(srs_path, settings_path)

    # now generate the witness file
    res = ezkl.gen_witness(data_path, compiled_model_path, witness_path, settings_path = settings_path)
    assert os.path.isfile(witness_path)

    res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
        srs_path,
        settings_path,
    )

    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

    # GENERATE A PROOF


    res = ezkl.prove(
            witness_path,
            compiled_model_path,
            pk_path,
            proof_path,
            srs_path,
            "poseidon", # for aggregated EVM proof only ELSE 'evm'
            "single",
            settings_path,
        )
    print ('Successfully generate ZK Proof!')
    print(res)
    assert os.path.isfile(proof_path)

# gen_verifier(clean_model, target, True)
#
# sol_code_path = os.path.join('Verifier.sol')
# abi_path = os.path.join('Verifier.abi')
# res = ezkl.create_evm_verifier(
#                 vk_path,
#                 srs_path,
#                 settings_path,
#                 sol_code_path,
#                 abi_path
#             )
#
# assert res == True
# assert os.path.isfile(sol_code_path)

