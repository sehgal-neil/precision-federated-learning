## Neil Sehgal, April 2022
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import time
import collections
import tensorflow_federated as tff
import tensorflow as tf
from tensorflow.python.keras import backend as K
import nest_asyncio
import sys


## Parameters for the simulation
N = [2500] # Sample Size
R_0 = [.1] # Disease Prevalence Among Unx
PE = [.3] # Exposure Prevalence
SILOS = [1, 5, 25, 100, -1] # Every Device Contributes Separately, A Few Hospitals(5), Many Little Sites(25), Central Repository
EFFECT_SIZE = [5] # Effect Size/Direction


# Create dataframe with combination of all parameters
N, R_0, PE, SILOS, EFFECT_SIZE = pd.core.reshape.util.cartesian_product([N, R_0, PE, SILOS, EFFECT_SIZE])
df = pd.DataFrame(dict(N=N, R_0=R_0, PE=PE, SILOS=SILOS, EFFECT_SIZE=EFFECT_SIZE))
df.loc[df["SILOS"] == -1, "SILOS"] = df["N"]
df['N'] = df['N'].astype(int)
df['SILOS'] = df['SILOS'].astype(int)
df['manual_Odds_Ratio'] = 0.0
df['smf_Odds_Ratio'] = 0.0
df['TF_Odds_Ratio'] = 0.0
df['TFF_Odds_Ratio'] = 0.0

df_array = df.to_numpy()

# To repeat simulations multiple times
# repeat = 2
# df = pd.concat([df]*repeat, ignore_index=True)

N_idx = 0
R_0_idx = 1
PE_idx = 2
SILOS_idx = 3
EFFECT_SIZE_idx = 4
manual_Odds_Ratio_idx = 5
smf_Odds_Ratio_idx = 6
TF_Odds_Ratio_idx = 7
TFF_Odds_Ratio_idx = 8

def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.Dense(
          1,
          activation='sigmoid',
          # activation='softmax',
          input_shape=(1,),
          kernel_regularizer=tf.keras.regularizers.l2(1e-6),
      )
  ])

"""Normal TF Regression"""

def tensorflow_regress(x, y):
  tf.keras.backend.clear_session()
  tf_model = create_keras_model()
  tf_model.compile(
              optimizer=tf.keras.optimizers.Nadam(learning_rate=0.15),   
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[
                      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                      tf.keras.metrics.AUC(name='auc'),
                      ]
              )
  tf_model.fit(x.reshape(-1,1), y, epochs=50, batch_size=100, verbose=0)
  return tf.exp(tf_model.get_weights()[0])

"""TFF Regression"""

"""
Takes in x and y numpy arrays and number of silos and returns federated data
"""
def get_client_dataset_train(data_train, labels_train, number_of_silos):
  data_train = data_train.reshape(-1, 1)
  datapoints_per_silo = int(np.floor(len(labels_train)/number_of_silos))
  client_train_dataset = collections.OrderedDict()

  for i in range(0, number_of_silos):
      # client_name = "client_" + str(i)
      client_name = i
      start = datapoints_per_silo * i
      end = datapoints_per_silo * (i+1)
      # print(f"Adding data from {start} to {end} for client : {client_name}")
      data = collections.OrderedDict(
          x=data_train[start:end],
          y=labels_train[start:end],
      )
      client_train_dataset[client_name] = data
  train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset)
  return train_dataset


def preprocess(dataset):
  return dataset.repeat(1).batch(1)

def make_federated_data(client_data, client_ids):
  return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]


def tff_regress(x, y, silos):
  print("Running TFF")
  K.clear_session()
  nest_asyncio.apply()
  tff.framework.set_default_context(tff.backends.native.create_thread_debugging_execution_context(clients_per_thread=50))
  # np.random.seed(10)
  # tf.random.set_seed(10)
  NUM_CLIENTS = silos
  NUM_ROUNDS = 450
  client_dataset_train = get_client_dataset_train(x, y, NUM_CLIENTS)
  preprocessed_example_dataset = preprocess(client_dataset_train.create_tf_dataset_for_client(client_dataset_train.client_ids[0]))
  
  def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.AUC(name='auc')])

  iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    server_optimizer_fn=lambda: tf.keras.optimizers.Nadam(learning_rate=0.5),
    use_experimental_simulation_loop = True
  )

  state = iterative_process.initialize()
  tff_model = create_keras_model()
  # step_size = max(NUM_CLIENTS//10, 1)

  # Use same federated train data each time
  federated_train_data = make_federated_data(client_dataset_train, np.random.choice(range(NUM_CLIENTS), size=NUM_CLIENTS, replace=False))

  for round_num in range(NUM_ROUNDS):
    state, metrics = iterative_process.next(state, federated_train_data)
    state.model.assign_weights_to(tff_model)
    # print(f"Participation: {participation}, Round Number: {round_num}, Metrics: {str(metrics)}")
  return tf.exp(tff_model.get_weights()[0])


def get_manual_OR(exposed, diseased):
  """
  Manual Odds Ratio calculation
  """
  exposed_and_diseased = np.sum(np.logical_and(exposed, diseased))
  unexposed_and_no_disease = np.sum(np.logical_and(np.logical_not(exposed), np.logical_not(diseased)))
  exposed_and_no_disease = np.sum(np.logical_and(exposed, np.logical_not(diseased)))
  unexposed_and_diseased = np.sum(np.logical_and(np.logical_not(exposed), diseased))
  odds_ratio = (exposed_and_diseased / exposed_and_no_disease) / (unexposed_and_diseased / unexposed_and_no_disease)
  return odds_ratio

def run_simulation(N, R_0, P_E, SILOS, EFFECT_SIZE):
  R_1 = EFFECT_SIZE*(R_0/(1-R_0)) / (1 + EFFECT_SIZE*(R_0/(1-R_0))) # Disease Prevalence among Exposed
  exposed = np.zeros((N))
  diseased = np.zeros((N))

  for j in range(1, N):
    exposed[j] = np.random.binomial(n = 1, p = P_E)

    if (exposed[j]==1): # exposed
      diseased[j] = np.random.binomial(n = 1, p = R_1)

    else: # unexposed
      diseased[j] = np.random.binomial(n = 1, p = R_0)
      

  df_dict2 = {"exposed": exposed, "diseased": diseased}
  df_2 = pd.DataFrame(df_dict2)
  exposed_logistic_model = smf.logit("diseased ~ exposed", data=df_2).fit()

  manual_or = get_manual_OR(exposed, diseased)
  smf_or_c = np.exp(exposed_logistic_model.params[1])
  tf_or_c = tensorflow_regress(x=exposed, y=diseased)
  tff_or_c = tff_regress(x=exposed, y=diseased, silos=SILOS)

  return (manual_or, smf_or_c, tf_or_c, tff_or_c)


def fill_row(index, value):
  print("INDEX: ", index)
  N = int(value[N_idx])
  R_0 = value[R_0_idx]
  P_E = value[PE_idx]
  SILOS = int(value[SILOS_idx])
  EFFECT_SIZE = value[EFFECT_SIZE_idx]

  manual_or, smf_or_c, tf_or_c, tff_or_c = run_simulation(N, R_0, P_E, SILOS, EFFECT_SIZE)

  value[manual_Odds_Ratio_idx] = manual_or
  value[smf_Odds_Ratio_idx] = smf_or_c 
  value[TF_Odds_Ratio_idx] = tf_or_c
  value[TFF_Odds_Ratio_idx] = tff_or_c


start_time = time.time()
for index, value in enumerate(df_array):
    fill_row(index, value)
execution_time = (time.time() - start_time)/60.0
print("Execution time (mins): ")
print(execution_time)


df[:] = df_array #Overwrite dataframe
file_name = 'simulation_' + sys.argv[1] + '.csv'
df.to_csv(file_name)
