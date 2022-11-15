import numpy as np
import pandas as pd
import math
import random
from numpy import sqrt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def entropy(data):
  """
  This function measures the amount of uncertainty in a probability distribution
  args: 
  * data(type: DataFrame): the data you're calculating for the entropy
  return:
  * entropy_value(type: float): the data's entropy
  """

  n_row, n_col = data.shape
  #print(n_row)
  data = data.iloc[:, n_col-1]
  

    # TF = data.iloc[ : ,9]
  one = np.sum(data)
  probs = (one / n_row)
  # print(probs)
  if (probs == 0 or probs == 1) : entropy_value = 0
  else : entropy_value = (-1)*((probs*np.log2(probs)) + ((1 - probs) * np.log2(1 - probs)))
  #print(probs)
  return entropy_value


def information_gain(data, mask):
  """
  This function will calculate the information gain
  args:
  * data(type: DataFrame): the data you're calculating for the information gain
  * mask(type: Series): partition information(left/right) of current input data, 
    - boolean 1(True) represents split to left subtree
    - boolean 0(False) represents split to right subtree
  return:
  * ig(type: float): the information gain you can obtain by classify data with this given mask
  """
  #print(split_boolean)
  # print(data[mask])
  # print(data[~mask])
  # print(entropy(data) ,entropy(data[mask]) ,entropy(data[~mask]))
  len_mask = len(mask)
  tf_prob = mask.value_counts()
  t_prob = tf_prob[True]/len_mask
  f_prob = tf_prob[False]/len_mask

  ig = entropy(data) - (entropy(data[mask])*t_prob + entropy(data[~mask])*f_prob)# now - last
  return ig


def find_best_split(data):
  """
  This function will find the best split combination of data
  args:
  * data(type: DataFrame): the input data
  return
  * best_ig(type: float): the best information gain you obtain
  * best_threshold(type: float): the value that splits data into 2 branches
  * best_feature(type: string): the feature that splits data into 2 branches
  """
  best_ig = 0
  best_threshold = 0
  best_feature = 0
  best_ig = float('-inf') #information gain
  n_rows, n_cols = data.shape

  for col in range(n_cols-1):
    attribute = data.columns[col]
    #print(attribute)
    sorted_data = data.sort_values(attribute)
    #print(sorted_data)
    for row in range(n_rows):
      if row == 0:
        continue
      elif sorted_data.iloc[row, col] == sorted_data.iloc[row-1, col]:
        #print("??")
        continue
      else:
        #print("?")
        mean_value = (sorted_data.iloc[row, col] + sorted_data.iloc[row-1, col]) / 2
        #print(mean_value)
        mask = sorted_data.iloc[:, col]
        split_tf = mask > mean_value
        #print(split_tf)
        ig = information_gain(sorted_data, split_tf)

        #print(attribute, ig, best_ig)
        if best_ig <= ig:
          best_feature = attribute
          best_ig = ig
          best_threshold = mean_value
  
  return best_ig, best_threshold, best_feature


def make_partition(data, feature, threshold):
  """
  This function will split the data into 2 branches
  args:
  * data(type: DataFrame): the input data
  * feature(type: string): the attribute(column name)
  * threshold(type: float): the threshold for splitting the data
  return:
  * left(type: DataFrame): the divided data that matches(less than or equal to) the assigned feature's threshold
  * right(type: DataFrame): the divided data that doesn't match the assigned feature's threshold
  """
  #print(threshold)
  mask = data[feature] > threshold
  right, left = data[mask], data[~mask]

  return left, right


def build_tree(data, max_depth, min_samples_split, depth):
  """
  This function will build the decision tree
  args:
  * data(type: DataFrame): the data you want to apply to the decision tree
  * max_depth: the maximum depth of a decision tree
  * min_samples_split: the minimum number of instances required to do partition
  * depth: the height of the current decision tree
  return:
  * subtree: the decision tree structure including root, branch, and leaf (with the attributes and thresholds)
  """
  n_col = len(data.columns)

  left_subtree, right_subtree = [], []
  # check the condition of current depth and the remaining number of samples
  if len(data) >= min_samples_split:
    if depth < max_depth:
      # call find_best_split() to find the best combination
      ig, threshold, feature = find_best_split(data)
      #print(ig)
      # check the value of information gain is greater than 0 or not 
      if ig > 0 :
        # update the depth
        depth = depth+1
        # call make_partition() to split the data into two parts
        left_subtree, right_subtree = make_partition(data, feature, threshold)
        # If there is no data split to the left tree OR no data split to the left tree
        if  len(left_subtree) == 0 or len(right_subtree) == 0:
          # return the label of the majority
          label = data.iloc[:, n_col-1].mode()[0]
          return label
        else:
          question = "{} {} {}".format(feature, "<=", threshold)
          subtree = {question: []}
          # print(left_subtree)
          # print(right_subtree)

          # call function build_tree() to recursively build the left subtree and right subtree
          l_subtree = build_tree(left_subtree, max_depth, min_samples_split, depth)
          #print(l_subtree.shape)
          r_subtree = build_tree(right_subtree, max_depth, min_samples_split, depth)
          #print(l_subtree.shape)
          if l_subtree == r_subtree:
            subtree = l_subtree
          else:
            subtree[question].append(l_subtree)
            subtree[question].append(r_subtree)
      else:
        # return the label of the majority
        label = data.iloc[:, n_col-1].mode()[0]
        return label
    else:
      # return the label of the majority
      label = data.iloc[:, n_col-1].mode()[0]
      return label
  else :
    label = data.iloc[:, n_col-1].mode()[0]
    return label

  return subtree



def classify_data(instance, tree):
  """
  This function will predict/classify the input instance
  args:
  * instance: a instance(case) to be predicted
  return:
  * answer: the prediction result (the classification result)
  """
  equation = list(tree.keys())[0] 
  if equation.split()[1] == '<=':
    temp_feature = equation.split()[0]
    temp_threshold = equation.split()[2]
    if instance[temp_feature] > float(temp_threshold):
      answer = tree[equation][1]
    else:
      answer = tree[equation][0]
  else:
    if instance[equation.split()[0]] in (equation.split()[2]):
      answer = tree[equation][0]
    else:
      answer = tree[equation][1]

  if not isinstance(answer, dict):
    return answer
  else:
    return classify_data(instance, answer)


def make_prediction(tree, data):
  """
  This function will use your pre-trained decision tree to predict the labels of all instances in data
  args:
  * tree: the decision tree
  * data: the data to predict
  return:
  * y_prediction: the predictions
  """
  y_prediction = []
  row = data.shape[0]
  for i in range(row):
    y_prediction.append(classify_data(data.loc[data.index[i]], tree))
  
  
  # [Note] You can call the function classify_data() to predict the label of each instance
  

  return y_prediction


def calculate_score(y_true, y_pred):
  """
  This function will calculate the f1-score of the predictions
  args:
  * y_true: the ground truth
  * y_pred: the predictions
  return:
  * score: the f1-score
  """
  score = f1_score(y_true, y_pred, average='binary')
  
  return score

def gen_tree(tree, index):
    """
    This function will return the direction of each node in the manner of graphivz.
    args:
    * tree: the decision tree
    * index: the index of the tree (represented in array)
    return:
    * out: the structure of tree
    """
    info = str(list(tree)[0]).split()
    child = list(tree.values())[0]

    temp = f'\t{index} [label = \"{info[0]}\\n{info[2]}\"];\n'
    temp += f'\t{index} -> {2 * index} [xlabel = \"<=\"];\n'
    temp += f'\t{index} -> {2 * index + 1} [xlabel = \">\"];\n'

    if isinstance(child[0], dict):
        temp += gen_tree(child[0], 2 * index)        
    else:
        temp += f'\t{2 * index} [label = \"{child[0]}\"];\n'

    if isinstance(child[1], dict):
        temp += gen_tree(child[1], 2 * index + 1)        
    else:
        temp += f'\t{2 * index + 1} [label = \"{child[1]}\"];\n'

    return temp
    
def graphviz_gen(tree, tree_name):
    """
    This function will turn the tree structure into graphviz style.
    args:
    * tree: the decision tree
    * tree_name: the name of the decision tree
    """
    result = f'digraph {tree_name}' + '{\n\tnode [shape = cube];\n'
    result += gen_tree(tree, 1)
    result += '}'

    with open(f'{tree_name}.dot', 'w') as f:
        f.write(result)


advanced_data = pd.read_csv('hw2_input_advanced.csv')


# You can split *advanced_data* into training set and validaiton set


training_data = []
validation_data = []


# ## Step2 : Load the test data
# Load the input file **hw2_input_test.csv** to make predictions with the pre-trained random forest model


x_test = pd.read_csv('hw2_input_test.csv')



# ## Step3 : Build a Random Forest


# Define the attributions of the random forest
# > * You **can** modify the values of these attributes in advanced part
# > * Each tree can have different attribute values
# > * There must be **at least** 3 decision trees in the random forest model
# > * Must use function *build_tree()* to build a random forest model
# > * These are the parameters you can adjust : 
# 
# 
#     ```
#     max_depth = 
#     depth = 0
#     min_samples_split = 
#     
#     # total number of trees in a random forest
#     n_trees = 
# 
#     # number of features to train a decision tree
#     n_features = 
# 
#     # the ratio to select the number of instances
#     sample_size = 
#     n_samples = int(training_data.shape[0] * sample_size)
#     ```
# 


# Define the attributes
n_input_data_rows, n_input_data_cols = advanced_data.shape

split_train_vaild_value = 0.9

max_depth = 3
depth = 0
min_samples_split = 6

# total number of trees in a random forest
n_trees = 11

# number of features to train a decision tree
n_features = 5

# split data into vaild and train

training_data = advanced_data.iloc[:int(n_input_data_rows*split_train_vaild_value), :]
validation_data = advanced_data.iloc[int(n_input_data_rows*split_train_vaild_value):, :]
y_validation = validation_data[["diabetes_mellitus"]]
x_validation = validation_data.drop(['diabetes_mellitus'], axis=1)
y_validation = y_validation.values.flatten()
    #print(training_data.shape, validation_data.shape)



# the ratio to select the number of instances
sample_size = 0.1
n_samples = int(training_data.shape[0] * sample_size)




def build_forest(data, n_trees, n_features, n_samples):
  #print(data.shape)
  """
  This function will build a random forest.
  args:
  * data: all data that can be used to train a random forest
  * n_trees: total number of tree
  * n_features: number of features
  * n_samples: number of instances
  return:
  * forest: a random forest with 'n_trees' of decision tree
  """
  forest = [] #list of tree
  tree_data = []
  for i in range(n_trees):
    tree_data = data.sample(n = n_samples)
    #print(tree_data.shape)
    #tf = tree_data.iloc[:, tree_data.shape[1]-1]
    tree_data.drop(['diabetes_mellitus'], axis=1)
    #print(tree_data.shape)
    tree_data.sample(n = n_features, axis=1)
    #print(tree_data.shape)
    #tree_data = pd.concat([tree_data, tf], axis=1)
    #print(tree_data.shape)
  # must reuse function build_tree()
    tree_data.index = range(len(tree_data))
    #print(tree_data)
    tree = build_tree(tree_data, max_depth, min_samples_split, depth)
    #print(tree)
    forest.append(tree)

  return forest


#forest = build_forest(training_data, n_trees, n_features, n_samples)

# for i in forest:
#     print(i)


# ## Step4 : Make predictions with the random forest
# > Note: Please print the f1-score of the predictions of each decision tree


def make_prediction_forest(forest, data):
  """
  This function will use the pre-trained random forest to make the predictions
  args:
  * forest: the random forest
  * data: the data used to predict
  return:
  * y_prediction: the predicted results
  """

  tf = []
  for tree in forest :
    p = []
    prediction = 0
    dt_size = len(data)
    # print(data.iloc[0,:])
    for row in range(dt_size) :
      prediction = classify_data(data.loc[data.index[row]], tree)
      p.append(prediction)
    tf.append(p)

  tf = np.transpose(tf)
  # print(TF_tmp)
  y_prediction = []

  for i in tf :
    for j in i :
      t = 0
      f = 0
      if (j == True) : t += 1
      else : f += 1
    if (t >= f) : y_prediction.append(True)
    else : y_prediction.append(False)


  return y_prediction




tmp_f1 = 0
tmp_forest = []

for i in range(100) :
    forest = build_forest(training_data, n_trees, n_features, n_samples)
    y_pred_test = make_prediction_forest(forest, validation_data)
    # print(y_pred_test)
    valid_tf = validation_data.iloc[:,len(validation_data.columns)-1]
    # print(valid_tf)
    valid_ans = calculate_score(valid_tf, y_pred_test)
    if (valid_ans > tmp_f1) : 
        tmp_f1 = valid_ans
        tmp_forest = forest
    print("f1_score : ", valid_ans) 

y_pred_test = make_prediction_forest(tmp_forest, validation_data)
# print(y_pred_test)
valid_tf = validation_data.iloc[:,len(validation_data.columns)-1]
# print(valid_tf)
valid_ans = calculate_score(valid_tf, y_pred_test)
print("f1_score : ", valid_ans)


y_pred_test = make_prediction_forest(tmp_forest, x_test)
print(len(y_pred_test))
# y_pred_test

# ## Step5 : Write the Output File
# Save your predictions from the **random forest** in a csv file, named as **hw2_advanced.csv**


advanced = []
for i in range(len(y_pred_test)):
  advanced.append(y_pred_test[i])


advanced_path = 'hw2_advanced.csv'
pd.DataFrame(advanced).to_csv(advanced_path, header = None, index = None)


