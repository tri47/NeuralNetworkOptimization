import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import mlrose

from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate, train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

def return_stratified_kcv_results(clf, x_data, y_data, verbose = False, last_curve = False):
   # x_data.reset_index(inplace=True)
    #sm = SMOTE(random_state=12, ratio = 0.3)
    #x_data, y_data = sm.fit_sample(x_data, y_data)

  #  unique, counts = np.unique(y_data, return_counts=True)
   # print (np.asarray((unique, counts)).T)

    y_data = y_data.to_list()
    y_data = np.array([y_data]).transpose()
   # print(np.shape(y_data))
  #  y_data = y_data.to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    train_scores, test_scores, train_accuracys, test_accuracys = [], [], [], []
    FPs = []
    FNs = []
    train_times, test_times = [], []
    curves = []
    for train_index, test_index in skf.split(x_data, y_data):
        print('a CV')
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
       
        if False:
            sm = SMOTE(random_state=12, ratio = 0.5)
            x_train, y_train = sm.fit_sample(x_train, y_train)
            unique, counts = np.unique(y_train, return_counts=True)
            print (np.asarray((unique, counts)).T)
            y_train = np.array([y_train]).transpose()
            print(np.shape(y_train))
        
        if False:
          #  cc = ClusterCentroids(random_state=0)
           # cc = NearMiss(version=3, n_neighbors=5)
           # cc = CondensedNearestNeighbour(n_neighbors=1)

           # cc = TomekLinks()
           # cc = EditedNearestNeighbours(n_neighbors=3)
            cc = OneSidedSelection(n_neighbors=1, n_seeds_S=200)
            x_train, y_train = cc.fit_resample(x_train, y_train)
            unique, counts = np.unique(y_train, return_counts=True)
            print (np.asarray((unique, counts)).T)
            y_train = np.array([y_train]).transpose()
            print(np.shape(y_train))            

        start_time = time.time()
        results = clf.fit(x_train, y_train)
        train_times.append(time.time()-start_time)
        y_train_pred = clf.predict(x_train)
        start_time = time.time()
        y_test_pred = clf.predict(x_test)
        test_times.append(time.time()-start_time)
       
        a_curve = results.fitness_curve
        curves.append(a_curve)

        a = np.concatenate([y_train,y_train_pred],axis=1)
        pd.DataFrame(a).to_csv("file.csv")
        train_score =f1_score(y_train, y_train_pred) 
        test_score =f1_score(y_test, y_test_pred) 
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        FP = confusion_matrix(y_train,y_train_pred, normalize='true')[0][1]
        FN = confusion_matrix(y_train,y_train_pred, normalize='true')[1][0]

        train_scores.append(train_score)
        test_scores.append(test_score)
        train_accuracys.append(train_accuracy)
        test_accuracys.append(test_accuracy)
        FPs.append(FP)
        FNs.append(FN)



    if last_curve:
        curves = curves[-1]
    else:
        curves = np.array(curves)
        curves = curves.mean(axis=0)
        print(np.shape(curves))
    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)
    train_accuracys = np.array(train_accuracys)
    test_accuracys = np.array(test_accuracys)
    train_times = np.array(train_times)
    test_times = np.array(test_times)     
    FPs = np.array(FPs)
    FNs = np.array(FNs)

    return curves, train_scores.mean(), test_scores.mean(), train_accuracys.mean(), \
        test_accuracys.mean(),train_times.mean(),test_times.mean(), \
        FPs.mean(), FNs.mean()

def run_random_hill_experiment_1(x_data,y_data, hidden_nodes):
    train_scores = [] 
    test_scores = [] 
    train_accs, test_accs = [], []
    learning_rates = [0.001, 0.01, 0.1, 0.2,  0.5, 1]
    
    for learning_rate in learning_rates:
        print(learning_rate)
        clf = mlrose.NeuralNetwork(
            hidden_nodes = hidden_nodes, activation = 'relu', \
            algorithm = 'random_hill_climb', max_iters = 1000, \
            bias = True, is_classifier = True, learning_rate = learning_rate, \
            early_stopping = True, clip_max = 5, max_attempts = 100, \
            random_state = 30, restarts = 0)
        curve, train_score, test_score, train_acc, test_acc, train_time, test_time, FP, FN = \
                return_stratified_kcv_results(clf, x_data, y_data)
        train_scores.append(train_score)
        test_scores.append(test_score)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
   
    plt.figure(1)
    plt.plot(learning_rates, train_scores)
    plt.plot(learning_rates, test_scores)
    plt.plot(learning_rates, test_accs)
    plt.legend(['train F1 score', 'CV F1 score ', 'CV Accuracy score'])
    plt.title('Random Hill Climb performance vs learning rates')
    plt.xlabel('Learning rate',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.show()

def run_random_hill_experiment_2(x_data,y_data, hidden_nodes):
    train_scores = [] 
    test_scores = [] 
    train_accs, test_accs = [], []
    train_times = []
    restarts = [0, 1, 10, 100]
    FPs, FNs = [] , []

    random_state = 0
    for restart in restarts:
        print(restart)
        clf = mlrose.NeuralNetwork(
            hidden_nodes = hidden_nodes, activation = 'relu', \
            algorithm = 'random_hill_climb', max_iters = 1000, \
            bias = True, is_classifier = True, learning_rate = 0.2, \
            early_stopping = True, clip_max = 5, max_attempts = 100, \
            random_state = None, restarts = restart, curve = True)
        curves, train_score, test_score, train_acc, test_acc, train_time, test_time, FP, FN = \
                return_stratified_kcv_results(clf, x_data, y_data)
        train_scores.append(train_score)
        test_scores.append(test_score)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_times.append(train_time)
        FPs.append(FP)
        FNs.append(FN)
    print(FPs)
    print(FNs)
   
    plt.figure(2)
    plt.plot(restarts, train_scores)
    plt.plot(restarts, test_scores)
    plt.plot(restarts, test_accs)
    plt.legend(['train F1 score', 'CV F1 score ', 'CV Accuracy score'])
    plt.title('Random Hill Climb performance vs restarts')
    plt.xlabel('Restarts',fontsize=12)
    plt.ylabel('Score',fontsize=12)

    plt.figure(3)
    plt.plot([i for i in range(0,len(curves))], curves)
    plt.title('Random Hill Climb fitness at each iteration')
    plt.xlabel('Iteration',fontsize=12)
    plt.ylabel('Fitness',fontsize=12)
    plt.show()

    plt.figure(4)
    plt.plot(restarts, train_times)
    plt.title('Random Hill Climb train time')
    plt.xlabel('Restarts',fontsize=12)
    plt.ylabel('Time (s)',fontsize=12)
    plt.show()

    plt.figure(5)
    plt.plot(restarts, FPs)
    plt.plot(restarts, FNs)
    plt.legend(['Type 1 Error (FP)', 'Type 2 Error (FN)'])
    plt.title('Random Hill Climb TYPE 1 (FP) and TYPE 2 (FN) Errors')
    plt.xlabel('Restarts',fontsize=12)
    plt.ylabel('Error Rate',fontsize=12)
    plt.show()

def run_SA_experiment_1(x_data,y_data, hidden_nodes):
    train_scores = [] 
    test_scores = [] 
    train_accs, test_accs = [], []
    learning_rates = [0.001, 1, 2, 5]

    print('Hidden nodes :', hidden_nodes)

    for learning_rate in learning_rates:
        print(learning_rate)
        clf = mlrose.NeuralNetwork(
            hidden_nodes = hidden_nodes, activation = 'relu', \
            algorithm = 'simulated_annealing', max_iters = 1000, \
            bias = True, is_classifier = True, learning_rate = learning_rate, \
            early_stopping = True, clip_max = 5, max_attempts = 100, \
            random_state = 30, restarts = 0)
        curves, train_score, test_score, train_acc, test_acc, train_time, test_time, FP, FN = \
                return_stratified_kcv_results(clf, x_data, y_data)
        print(train_score)
        train_scores.append(train_score)
        test_scores.append(test_score)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
   
    plt.figure(1)
    plt.plot(learning_rates, train_scores)
    plt.plot(learning_rates, test_scores)
    plt.plot(learning_rates, test_accs)
    plt.legend(['train', 'test', 'acc'])
    plt.show()

def run_SA_experiment_2(x_data,y_data, hidden_nodes):
    train_scores = [] 
    test_scores = [] 
    train_accs, test_accs = [], []
    train_times = []
    iters = [100, 500, 1000, 5000, 10000]
    FPs, FNs = [] , []

    for iter in iters:
        print(iter)
        clf = mlrose.NeuralNetwork(
            hidden_nodes = hidden_nodes, activation = 'relu', \
            algorithm = 'simulated_annealing', max_iters = iter, \
            bias = True, is_classifier = True, learning_rate = 0.2, \
            early_stopping = True, clip_max = 5, max_attempts = 100, \
            random_state = 30, restarts = 0, curve = True)
        curves, train_score, test_score, train_acc, test_acc, train_time, test_time, FP, FN = \
                return_stratified_kcv_results(clf, x_data, y_data)

        train_scores.append(train_score)
        test_scores.append(test_score)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_times.append(train_time)   
        FPs.append(FP)
        FNs.append(FN)
    print(FPs)
    print(FNs)

    plt.figure(4)
    plt.plot(iters, train_scores)
    plt.plot(iters, test_scores)
    plt.plot(iters, test_accs)
    plt.legend(['train F1 score', 'CV F1 score ', 'CV Accuracy score'])
    plt.title('Simulated Annealing performance vs max iterations')
    plt.xlabel('Max iterations',fontsize=12)
    plt.ylabel('Score',fontsize=12)

    plt.figure(5)
    plt.plot([i for i in range(0,len(curves))], curves)
    plt.title('Simulated Annealing fitness at each iteration')
    plt.xlabel('Iteration',fontsize=12)
    plt.ylabel('Fitness',fontsize=12)
    plt.show()

    plt.figure(6)
    plt.plot(iters, train_times)
    plt.title('Simulated Annealing train time')
    plt.xlabel('Iteration',fontsize=12)
    plt.ylabel('Time (s)',fontsize=12)
    plt.show()

    plt.figure(7)
    plt.plot(iters, FPs)
    plt.plot(iters, FNs)
    plt.legend(['Type 1 Error (FP)', 'Type 2 Error (FN)'])
    plt.title('Simulated Annealing TYPE 1 (FP) and TYPE 2 (FN) Error Rates')
    plt.xlabel('Max iterations',fontsize=12)
    plt.ylabel('Error Rate',fontsize=12)
    plt.show()

def run_GA_experiment_1(x_data,y_data, hidden_nodes):
    train_scores = [] 
    test_scores = [] 
    train_accs, test_accs = [], []
    train_times = []
    iters = [100, 500, 1000, 5000, 10000]
    FPs, FNs = [] , []

    for iter in iters:
        print(iter)
        clf = mlrose.NeuralNetwork(
            hidden_nodes = hidden_nodes, activation = 'relu', \
            algorithm = 'genetic_alg', max_iters = iter, \
            bias = True, is_classifier = True, learning_rate = 0.2, \
            early_stopping = True, clip_max = 5, max_attempts = 100, \
            random_state = 30, pop_size =  100, mutation_prob = 0.1, curve = True)
        curves, train_score, test_score, train_acc, test_acc, train_time, test_time, FP, FN = \
                return_stratified_kcv_results(clf, x_data, y_data, last_curve= True)

        train_scores.append(train_score)
        test_scores.append(test_score)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_times.append(train_time)   
        FPs.append(FP)
        FNs.append(FN)
    print(FPs)
    print(FNs)

    plt.figure(1)
    plt.plot(iters, train_scores)
    plt.plot(iters, test_scores)
    plt.plot(iters, test_accs)
    plt.legend(['train F1 score', 'CV F1 score ', 'CV Accuracy score'])
    plt.title('Genetic Algorithm performance vs max iterations')
    plt.xlabel('Max iterations',fontsize=12)
    plt.ylabel('Score',fontsize=12)

    plt.figure(2)
    plt.plot([i for i in range(0,len(curves))], curves)
    plt.title('Genetic Algorithm fitness at each iteration')
    plt.xlabel('Iteration',fontsize=12)
    plt.ylabel('Fitness',fontsize=12)
    plt.show()

    plt.figure(3)
    plt.plot(iters, train_times)
    plt.title('Genetic Algorithm train time')
    plt.xlabel('Iteration',fontsize=12)
    plt.ylabel('Time (s)',fontsize=12)
    plt.show()

    plt.figure(4)
    plt.plot(iters, FPs)
    plt.plot(iters, FNs)
    plt.legend(['Type 1 Error (FP)', 'Type 2 Error (FN)'])
    plt.title('Genetic Algorithm TYPE 1 (FP) and TYPE 2 (FN) Error Rates')
    plt.xlabel('Max iterations',fontsize=12)
    plt.ylabel('Error Rate',fontsize=12)
    plt.show()

def run_GA_experiment_2(x_data,y_data, hidden_nodes):
    train_scores = [] 
    test_scores = [] 
    train_accs, test_accs = [], []
    train_times = []
    pop_sizes = [50, 100, 200, 500]

    for pop_size in pop_sizes:
        print('pop size: ', pop_size)
        clf = mlrose.NeuralNetwork(
            hidden_nodes = hidden_nodes, activation = 'relu', \
            algorithm = 'genetic_alg', max_iters = 100, \
            bias = True, is_classifier = True, learning_rate = 0.2, \
            early_stopping = True, clip_max = 5, max_attempts = 100, \
            random_state = 30, pop_size =  pop_size, mutation_prob = 0.1, curve = True)
        curves, train_score, test_score, train_acc, test_acc, train_time, test_time, FP, FN = \
                return_stratified_kcv_results(clf, x_data, y_data)

        train_scores.append(train_score)
        test_scores.append(test_score)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_times.append(train_time)   

    plt.figure(4)
    plt.plot(pop_sizes, train_scores)
    plt.plot(pop_sizes, test_scores)
    plt.plot(pop_sizes, test_accs)
    plt.legend(['train F1 score', 'CV F1 score ', 'CV Accuracy score'])
    plt.title('Genetic Algorithm performance vs population size')
    plt.xlabel('Population sizes',fontsize=12)
    plt.ylabel('Score',fontsize=12)

    plt.figure(5)
    plt.plot([i for i in range(0,len(curves))], curves)
    plt.title('Genetic Algorithm fitness at each iteration')
    plt.xlabel('Iteration',fontsize=12)
    plt.ylabel('Fitness',fontsize=12)
    plt.show()

    plt.figure(6)
    plt.plot(pop_sizes, train_times)
    plt.title('Genetic Algorithm train time')
    plt.xlabel('population sizes',fontsize=12)
    plt.ylabel('Time (s)',fontsize=12)
    plt.show()

def run_GA_experiment_3(x_data,y_data, hidden_nodes):
    train_scores = [] 
    test_scores = [] 
    train_accs, test_accs = [], []
    train_times = []
    pop_sizes = [50, 2000]

    print('Hidden nodes :', hidden_nodes)

    for pop_size in pop_sizes:
        clf = mlrose.NeuralNetwork(
            hidden_nodes = hidden_nodes, activation = 'relu', \
            algorithm = 'genetic_alg', max_iters = 200, \
            bias = True, is_classifier = True, learning_rate = 0.2, \
            early_stopping = True, clip_max = 5, max_attempts = 100, \
            random_state = 30, pop_size =  pop_size, mutation_prob = 0.1, curve = True)
        curves, train_score, test_score, train_acc, test_acc, train_time, test_time, FP, FN = \
                return_stratified_kcv_results(clf, x_data, y_data, last_curve=True)

        train_scores.append(train_score)
        test_scores.append(test_score)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_times.append(train_time)   

    plt.figure(7)
    plt.plot(pop_sizes, train_scores)
    plt.plot(pop_sizes, test_scores)
    plt.plot(pop_sizes, test_accs)
    plt.legend(['train F1 score', 'CV F1 score ', 'CV Accuracy score'])
    plt.title('Genetic Algorithm performance vs population size')
    plt.xlabel('Population sizes',fontsize=12)
    plt.ylabel('Score',fontsize=12)

    plt.figure(8)
    plt.plot([i for i in range(0,len(curves))], curves)
    plt.title('Genetic Algorithm fitness at each iteration')
    plt.xlabel('Iteration',fontsize=12)
    plt.ylabel('Fitness',fontsize=12)
    plt.show()

    plt.figure(9)
    plt.plot(pop_sizes, train_times)
    plt.title('Genetic Algorithm train time')
    plt.xlabel('population sizes',fontsize=12)
    plt.ylabel('Time (s)',fontsize=12)
    plt.show()

# ********************************************
# ************ RUN EXPERIMENTS ***************

def main(alg):

    # PREPROCESS WILT DATA
    data = pd.read_csv('wilt_full.csv')
    data['class'].replace(['n'],0,inplace=True)
    data['class'].replace(['w'],1,inplace=True)
    x_data = data.loc[:, data.columns != 'class']
    y_data = data.loc[:,'class']
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)
    random_state = 100
    x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.3, random_state=random_state, shuffle=True, stratify=y_data)

    if alg == 'rhc':
        run_random_hill_experiment_1(x_train, y_train, [6,6])
        run_random_hill_experiment_2(x_train, y_train, [6,6])
    elif alg == 'sa':
        run_SA_experiment_1(x_train, y_train, [6,6])
        run_SA_experiment_2(x_train, y_train, [6,6])
    elif alg == 'ga':
        run_GA_experiment_1(x_train, y_train, [6,6])
        run_GA_experiment_2(x_train, y_train, [6,6])
        run_GA_experiment_3(x_train, y_train, [6,6])

if __name__ == "__main__" :
    import argparse
    print("Running Supervised Learning Experiments")
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', default='rhc')

    args = parser.parse_args()
    alg = args.alg
    if alg == 'rhc':
        print("Running Random Hill Climbing:...")
    if alg == 'sa':
        print("Running Simulated Annealing:...")
    if alg == 'ga':
        print("Running Genetic Algorithm:...")
    main(alg)
  




