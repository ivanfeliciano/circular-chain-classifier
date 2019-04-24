#!/usr/bin/env python
# coding: utf-8

# In[14]:


import math
import json

#Para leer archivo arfff
from scipy.io import arff
#Para manipular los conjuntos de datos más fácil
import pandas as pd
#Para medir la precisión del clasificador
from sklearn.metrics import accuracy_score


# ### Carga los conjuntos de datos flags-train y flags-test 

# In[15]:


# Lee el archivo .arff y lo carga a algo parecido a un diccionario
data_flag = arff.loadarff('./flags/flags-train.arff')
# Crea el dataframe del conjunto de entrenamiento
train_df_flags = pd.DataFrame(data_flag[0])
# Lo mismo pero con el conjunto de prueba
data_flag = arff.loadarff('./flags/flags-test.arff')
test_df_flags = pd.DataFrame(data_flag[0])
# Descomentar si sólo se quieren unas cuantas instancias
#test_df_flags = test_df_flags.head(2)


# ### Carga los conjuntos de datos emotions-train y emotions-test 

# In[16]:


data_emotions = arff.loadarff('./emotions/emotions-train.arff')
train_df_emo = pd.DataFrame(data_emotions[0])
data_emotions = arff.loadarff('./emotions/emotions-test.arff')
test_df_emo = pd.DataFrame(data_emotions[0])


# ### Funciones para hacer la discretización de variables continuas usando PKID

# In[17]:


global inf
inf = 10**20

def different(prev, current):
    eps=10e-5
    if prev != inf:
        return abs(prev-current) > eps
    else:
        return True

def proportional_k_interval_discretization(df):
    """
    Aplica dicretización intervalo k proporcional a un dataframe de
    pandas.
    """
    n = len(df)
    n_sqrt = int(math.sqrt(n))
    for attribute in df:
        if df[attribute].dtype != 'object':
            df = df.sort_values(attribute)
            local_index = 0
            current_interval = 0
            pred = inf
            for index, row in df.iterrows():
                #el intervalo incrementa cada sqrt(n) 
                #el tamaño si debe crecer
                local_index %= n_sqrt
                pred = df.at[index, attribute]
                df.at[index, attribute] = current_interval
                if local_index == n_sqrt - 1 and different(pred,df.at[index,attribute]) and current_interval < n_sqrt-1:
                    current_interval += 1           
                if not (local_index == n_sqrt-1 and not different(pred,df.at[index,attribute])):
                    local_index += 1                   
    return df


# ### Implementación del entrenador de Bayes

# In[18]:


def train_bayes_freq(labels, dataframe, output_file="freqs.json"):
    """
    Función para crear un JSON con las frecuencias de los atributos para después
    calcular las probabilidades condicionales y a priori.
    
    Dado un conjunto de datos como el siguiente:
    
    +----------+------+----------+----------+----------+----------+------+---------+------+------+
    | landmass | zone | language | religion | crescent | triangle | icon | animate | text | red  |
    +----------+------+----------+----------+----------+----------+------+---------+------+------+
    | b'5'     | b'1' | b'10'    | b'7'     | b'0'     | b'0'     | b'0' | b'0'    | b'0' | b'0' |
    +----------+------+----------+----------+----------+----------+------+---------+------+------+
    | b'6'     | b'1' | b'1'     | b'1'     | b'0'     | b'0'     | b'1' | b'1'    | b'1' | b'1' |
    +----------+------+----------+----------+----------+----------+------+---------+------+------+
    | b'5'     | b'1' | b'8'     | b'2'     | b'0'     | b'0'     | b'0' | b'0'    | b'0' | b'1' |
    +----------+------+----------+----------+----------+----------+------+---------+------+------+
    | b'5'     | b'1' | b'8'     | b'2'     | b'0'     | b'0'     | b'0' | b'0'    | b'0' | b'1' |
    +----------+------+----------+----------+----------+----------+------+---------+------+------+
    
    Se genera el JSON:
    
    {
      "language": {
        "b'10'": 1,
        "b'2'": 1,
        "b'8'": 1,
        "numberOfClasses": 4,
        "b'1'": 1,
        "red": {
          "b'10'": {
            "b'1'": 1
          },
          "b'1'": {
            "b'1'": 1
          },
          "b'2'": {
            "b'0'": 1
          },
          "b'8'": {
            "b'1'": 1
          }
        }
      },
      "triangle": {
        "numberOfClasses": 2,
        "b'1'": 1,
        "b'0'": 3,
        "red": {
          "b'1'": {
            "b'1'": 1
          },
          "b'0'": {
            "b'0'": 1
          }
        }
      },
      "text": {
        "numberOfClasses": 1,
        "b'0'": 4,
        "red": {
          "b'0'": {
            "b'0'": 1
          }
        }
      },
      "zone": {
        "b'4'": 3,
        "numberOfClasses": 2,
        "b'1'": 1,
        "red": {
          "b'4'": {
            "b'0'": 1
          },
          "b'1'": {
            "b'1'": 1
          }
        }
      },
      "landmass": {
        "b'4'": 2,
        "numberOfClasses": 2,
        "b'1'": 2,
        "red": {
          "b'4'": {
            "b'1'": 2
          },
          "b'1'": {
            "b'1'": 1
          }
        }
      },
      "crescent": {
        "numberOfClasses": 2,
        "b'1'": 1,
        "b'0'": 3,
        "red": {
          "b'1'": {
            "b'1'": 1
          },
          "b'0'": {
            "b'0'": 1
          }
        }
      },
      "N": 4,
      "icon": {
        "numberOfClasses": 1,
        "b'0'": 4,
        "red": {
          "b'0'": {
            "b'0'": 1
          }
        }
      },
      "religion": {
        "b'2'": 1,
        "b'5'": 1,
        "numberOfClasses": 4,
        "b'1'": 1,
        "b'0'": 1,
        "red": {
          "b'5'": {
            "b'1'": 1
          },
          "b'1'": {
            "b'1'": 1
          },
          "b'2'": {
            "b'1'": 1
          },
          "b'0'": {
            "b'0'": 1
          }
        }
      },
      "animate": {
        "numberOfClasses": 2,
        "b'1'": 1,
        "b'0'": 3,
        "red": {
          "b'1'": {
            "b'1'": 1
          },
          "b'0'": {
            "b'0'": 1
          }
        }
      },
      "red": {
        "numberOfClasses": 2,
        "b'1'": 3,
        "b'0'": 1
      }
    }
    
    :param labels: Un arreglo con el nombre de las etiquetas u objetivos del conjunto de datos.
    :type labels: list.
    :param dataframe: El dataframe de pandas con el conjunto de datos de entrenamiento.
    :type dataframe: pandas.Dataframe.
    
    """
    
    nc = 'numberOfClasses'
    len_training_instances = 'N'
    
    # El diccionario donde se almacenan las frecuencias 
    # y que se guarda en un JSON al final de la función
    
    frequency = dict()
    frequency[len_training_instances] = len(dataframe)
    
    # Por cada atributo objetivo (etiqueta) contamos frecuencias
    for label in labels:
        
        freq_label = dataframe[label].value_counts()
        
        # Si la etiqueta no está en el diccionario la agregamos y 
        # además añadimos el número de clases distintas
        if not label in frequency:
            frequency[label] = dict()
            frequency[label][nc] = len(dataframe[label].unique())
        # Iteramos sobre todos los posibles valores de la clase y sus 
        # frecuencias
        for label_val, freq in freq_label.iteritems():
            #casted_label_val = label_val.decode('ASCII') if type(label_val) is bytes else label_val
            casted_label_val = str(label_val)
            # Guardamos en el diccionarios los nombres de las clases de nuestra etiqueta y
            # sus frecuencias
            frequency[label][casted_label_val] = freq
        
        # Iteramos sobre todas las instancias de entrenamiento.
        for attribute in dataframe:
            if attribute == label:
                continue
            # Si no existe el atributo en nuestro diccionario de frecuencias.
            if not attribute in frequency:
                frequency[attribute] = dict()
                # El número de valores distintos que puede tomar el atributo
                frequency[attribute][nc] =len(dataframe[attribute].unique())
                
                # Por cada valor distinto que puede tomar el atributo,
                # sacamos la frecuencia.
                freq_attr = dataframe[attribute].value_counts()
                for att_val, freq in freq_attr.iteritems():
                    #cast_att_val = att_val.decode('ASCII') if type(att_val) is bytes else att_val
                    cast_att_val = str(att_val)
                    frequency[attribute][cast_att_val] = freq
            
            # Calculas las frecuencias por cada valor que toma el atributo 
            # y cada valor que puede tomar la etiqueta. Esto es X_i = x_i AND C = c
            frequency[attribute][label] = dict()
            freq_attr_and_label_series = dataframe.groupby(attribute)[label].value_counts()
            for index, value in freq_attr_and_label_series.iteritems():
                #attr_value = index[0].decode('ASCII') if type(index[0]) is bytes else index[0]
                #label_value = index[1].decode('ASCII') if type(index[1]) is bytes else index[1]
                attr_value = str(index[0])
                label_value = str(index[1])
                frequency[attribute][label][attr_value] = dict()
                frequency[attribute][label][attr_value][label_value] = value

    # Guardo el diccionario como un JSON.
    with open(output_file, 'w') as outfile:
        json.dump(frequency, outfile)


def apply_bayes(label, dataframe, input_file="freqs.json", k=1, m=2):
    """
    Aplica Naive Bayes a un conjunto de prueba en un dataframe de pandas.
    
    
    """
    pred_labels = []
    key_nc = 'numberOfClasses'
    
    # Obtiene un objeto con los valores de la clase.
    y = dataframe[label].unique()
    
    # Abre el archivo con las frecuencias del conjunto de entrenamiento.
    with open(input_file, 'r') as f:
        freq = json.loads(f.read())
    
    # Define el N, el número de instancias.
    N = freq['N']
    
    
    # Se itera sobre cada instancia de prueba.
    for _, row in dataframe.iterrows():
        
        # Se incializa la variable que guarda la etiqueta de la instancia.
        row_label = None
        max_prob = -10E10
        
        # Por cada valor posible de la clase u etiqueta.
        for y_hat in y:
            #cast_y_hat = y_hat.decode('ASCII') if type(y_hat) is bytes else y_hat
            
            # Convertimos a cadena la etiqueta para poder consultarla en el 
            # diccionario de frecuencias.
            
            cast_y_hat = str(y_hat)
            
            # nc = # de instancias que satisfacen C = c
            nc = freq[label].get(cast_y_hat, 0)
            
            # n  = # de clases
            n = freq[label][key_nc]
            
            # Calculamos P(C=c) usando Laplace-estimate
            p_c_laplace_estimator = (nc + k) / (N + n * k)
            ans = p_c_laplace_estimator
            
            #print("P({}={}) = {}".format(label, cast_y_hat, ans))
            # Iteramos sobre cada atributo de la instancia para
            # calcular P(Xi=xi | C=c) usando M-estimate
            for attr, val in row.iteritems():
                if attr == label:
                    continue
                #x_i = val.decode('ASCII') if type(val) is bytes else val
                
                
                x_i = str(val)
                p_xi_laplace_estimator = 0
                
                # Obtenemos el # de instancias que satisfacen Xi = xi
                n_xi = freq[attr].get(x_i, 0)
                
                # Obtenemos el # de valores posibles que toma Xi
                n = freq[attr][key_nc]
                
                #print("({} + {}) / ({} + {} * {})".format(n_xi, k, N, n, k))
                
                # Calculamos P(Xi=xi) usando Laplace-estimate
                p_xi_laplace_estimator = (n_xi + k) / (N + n * k)
                nci = 0
                #print("P({}={}) = {}".format(attr, x_i, p_xi_laplace_estimator))
                
                # Obtenemos el número de instancias que satisdacen Xi = xi y C = c
                if x_i in freq[attr][label] and cast_y_hat in freq[attr][label][x_i]:
                    nci = freq[attr][label][x_i][cast_y_hat]
                #print("({} + {} * {}) / ({} + {})".format(nci, m, p_xi_laplace_estimator, nc, m))
                
                # Calculamos P(Xi=xi | C=c) usando M-estimate
                m_estimator_xi_given_c = (nci + m * p_xi_laplace_estimator) / (nc + m)
                
                #print("P({}={}|{}={}) = {}".format(attr, x_i, label, cast_y_hat, m_estimator_xi_given_c))
                
                # Hacemos el producto de la probabilidades
                #ans *= m_estimator_xi_given_c
            # Cambio la etiqueta si la probabilidad es mayor para este valor de la 
            # etiqueta.
            if ans > max_prob:
                row_label = cast_y_hat
                max_prob = ans
        # Agrego a mi vector de etiquetas inferida
        # pred_labels.append("b'0'" if row_label == "b'1'" else "b'1'")
        pred_labels.append(row_label)
    correct_labels = [str(i) for i in list(dataframe[label])]
    print("Accuracy Score = {}".format(accuracy_score(correct_labels, pred_labels)))
    print("Accuracy Score Not Normalized = {}".format(accuracy_score(correct_labels, pred_labels, normalize=False)))


# In[19]:


labels = ['red','green','blue','yellow','white','black','orange']
train_df_flags = proportional_k_interval_discretization(train_df_flags)
test_df_flags = proportional_k_interval_discretization(test_df_flags)
train_bayes_freq(labels, train_df_flags)


# In[20]:


for lab in labels:
    apply_bayes(lab, test_df_flags)


# In[8]:


labels = ['amazed-suprised', 'happy-pleased', 'relaxing-calm', 'quiet-still', 'sad-lonely', 'angry-aggresive']
train_df_emo = proportional_k_interval_discretization(train_df_emo)
test_df_emo = proportional_k_interval_discretization(test_df_emo)
train_bayes_freq(labels, train_df_emo, 'train_emo_freq.json')


# In[9]:


for lab in labels:
    apply_bayes(lab, test_df_emo, 'train_emo_freq.json')


# In[ ]:




