from flask import Flask, send_file, make_response
from flask import render_template
from flask import request
import json
import io
import pandas 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.externals import joblib 





app = Flask(__name__)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/diagnosis', methods=['GET', 'POST'])
def diagnosis():
    if request.method == 'POST':
        loadDataToFile(request)
        dataframe = CreateDataframe()
        (df, data, target) = preProcess(dataframe)
        (keyValue, names, results) = machineLearning(data, target)
        responseJSON = transformToJSON(keyValue)
        return responseJSON
 
def loadDataToFile(request):
    data = request.get_json()
    content = data['content']
    try:
        f = open("data.csv", "w")
        f.write(content)
        f.close()
    except:
        print("error creating and writing file")
        
def CreateDataframe():    
    dataframe = pandas.read_csv("data.csv")
    return dataframe

def getColumnsNames(dataframe):
    return dataframe.columns.values

def preProcess(dataframe):
    columnsNames = getColumnsNames(dataframe)
    numberOfColumns = len(columnsNames)    
    array = dataframe.values
    print(numberOfColumns)
    selected = []
    for i in range(numberOfColumns-1): 
        selected.append(i)
    data = array[:,selected]
    target = array[:,numberOfColumns-1]
    df = dataframe.drop([dataframe.columns[numberOfColumns-1]], axis='columns')
    print(df.describe())
    return (df, data, target)

def machineLearning(data, target):
    print ("starting ML")
    # prepare configuration for cross validation test harness
    seed = 7
    # prepare models
    models = []
    models.append(('LR', LogisticRegression(solver='lbfgs')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(("QDA", QuadraticDiscriminantAnalysis()))
    models.append(('"KNN"', KNeighborsClassifier(3)))
    models.append(("DecisionTree", DecisionTreeClassifier()))
    models.append(("NaiveBayes", GaussianNB()))
    models.append(("LinearSVM", SVC(kernel="linear", C=0.025)))
    models.append(("RBF_SVM", SVC(gamma=2, C=1)))
    models.append(("GaussianProcess", GaussianProcessClassifier(1.0 * RBF(1.0))))
    models.append(("NeuralNet", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)))
    models.append(("RandomForest", RandomForestClassifier(max_depth=5, n_estimators=10)))
    models.append(("AdaBoost", AdaBoostClassifier()))
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=seed)
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    keyValue = {}
    msg = ''
    for name, model in models:
        #kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
        y_train_pred = cross_val_predict(model, X_train, y_train, cv=3)
        results.append(cv_results)
        names.append(name)
        keyValue[name] = [cv_results.mean(), cv_results.std(), str(confusion_matrix(y_train, y_train_pred))]
        #msg += "%s: %f %f " % (name, cv_results.mean(), cv_results.std())
        #msg += str(confusion_matrix(y_train, y_train_pred)) +"-\n"
        print(name + "done ********************")
    print (keyValue)
    return (keyValue, names, results)

def transformToJSON(keyValue):
    return json.dumps(keyValue)

@app.route('/comparison')
def algoCompare():
    # boxplot algorithm comparison
    dataframe = CreateDataframe()
    (df, data, target) = preProcess(dataframe)
    (keyValue, names, results) = machineLearning(data, target)    
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return send_file(bytes_image, attachment_filename='plot.png', mimetype='image/png')


@app.route('/histograma')
def histograma():
    dataframe = CreateDataframe()
    dataframe.hist(bins=50, figsize=(20,15))
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return send_file(bytes_image, attachment_filename='plot.png', mimetype='image/png')

@app.route('/correlation')
def correlation():
    dataframe = CreateDataframe()
    (df, data, target) = preProcess(dataframe)
    corrmat = df.corr()
    f, ax = plt.subplots(figsize =(9, 8)) 
    sns.heatmap(corrmat, ax = ax,  vmin = -1, vmax = 1, cmap ="YlGnBu", linewidths = 0.1)
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return send_file(bytes_image, attachment_filename='plot.png', mimetype='image/png')

@app.route('/pca')
def pca():
    dataframe = CreateDataframe()
    (df, data, target) = preProcess(dataframe)
    #pca to 2 elements and prepare graph
    pca = PCA(n_components=2, svd_solver='full')
    scaler=StandardScaler()#instantiate
    scaler.fit(data) # compute the mean and standard which will be used in the next command
    X_scaled=scaler.transform(data)# fit and transform can be applied together and I leave that for simple exercise
    pca.fit(X_scaled)
    principalComponents = pca.transform(X_scaled)
    principalDf = pandas.DataFrame(data = principalComponents, 
                                    columns = ['principal component 1', 'principal component 2'])
    targetDf = pandas.DataFrame(data = target, columns = ['target'])
    finalDf = pandas.concat([principalDf, targetDf], axis = 1)
    finalDf.plot(kind="scatter",title="Principal Component Analisys", x="principal component 1", y="principal component 2", 
                    alpha=0.6, c="target", cmap=plt.get_cmap("jet"), colorbar=True)
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return send_file(bytes_image, attachment_filename='plot.png', mimetype='image/png')

@app.route('/createModel', methods=['GET', 'POST'])
def createModel():
    if request.method == 'POST':
        data = request.get_json()
        researcherName = data['researcherName']
        algorithmName = data['algorithmName']

        dataframe = CreateDataframe()
        (df, data, target) = preProcess(dataframe)
        model = get_model(algorithmName)
        model.fit(data, target)

        myFileName = researcherName+algorithmName+'.pkl'
        # Save the model as a pickle in a file 
        joblib.dump(model, myFileName) 
        fileNameJSON = transformToJSON({'fileName': myFileName})
        return fileNameJSON

def get_model(algo):
    if algo == 'LR':
        return LogisticRegression(solver='lbfgs')
    elif algo == 'LDA':
        return LinearDiscriminantAnalysis()
    elif algo == 'QDA':
        return QuadraticDiscriminantAnalysis()
    elif algo == 'KNN':
        return KNeighborsClassifier(3)
    elif algo == 'DecisionTree':
        return DecisionTreeClassifier()
    elif algo == 'NaiveBayes':
        return GaussianNB()
    elif algo == 'LineatSVM':
        return SVC(kernel="linear", C=0.025)
    elif algo == 'RBF_SVM':
        return SVC(gamma=2, C=1)
    elif algo == 'GaussianProcess':
        return GaussianProcessClassifier(1.0 * RBF(1.0))
    elif algo == 'NeuralNet':
        return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    elif algo == 'RandomForest':
        return RandomForestClassifier(max_depth=5, n_estimators=10)
    elif algo == 'AdaBoost':
        return AdaBoostClassifier()
    else:
        return 'Algorithm does not exist, try again'

@app.route('/form', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        firstName = request.form.get('firstName')
        lastName = request.form.get('lastName')
        id = request.form.get('id')
        gender = request.form.get('gender')
        birthday = request.form.get('birthday')
        exam1 = request.form.get('exam1')
        exam2 = request.form.get('exam2')
        exam3 = request.form.get('exam3')
        parameters = request.form.get('parameters')
        modelName =request.form.get('modelName')
        diagnostic = request.form.get('diagnostic')

        print(type(firstName))
        print(type(lastName))
        print(type(id))
        print(type(gender))
        print(type(birthday))
        print(type(exam1))
        print(type(parameters))
        print(type(modelName))
        print(type(diagnostic))
        try:
            p = open("predict.csv", "w")
            p.write(parameters)
            p.close()
        except:
            print("error creating and writing file")

        names = ["meanGL", "SD-GL", "Smoothness", "ThirdMoment", "Uniformity", "Entropy"]
        dataframe = pandas.read_csv('predict.csv', names=names)
        array = dataframe.values
        data = array[:,:]

        prediction = ""
        try:
            my_model_loaded = joblib.load(modelName)
            prediction = my_model_loaded.predict(data)
        except FileNotFoundError as fnf_error:
            prediction = fnf_error

        response = {
                'prediction': str(prediction),
                'firstName': firstName,
                'lastName': lastName,
                'id': id,
                'gender': gender,
                'birthday': birthday,
                'exam1': exam1,
                'exam2': exam2,
                'exam3': exam3,
                'parameters': parameters,
                'modelName': modelName,
                'diagnostic': diagnostic
            }
        responseJSON = transformToJSON(response)
        print(responseJSON)
        return render_template('result.html', patient = response)

    
