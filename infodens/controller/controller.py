from infodens.feature_extractor import feature_manager as featman
from infodens.preprocessor.preprocess_services import Preprocess_Services
from infodens.classifier import classifier_manager
from infodens.formater import format
from infodens.controller.configurator import Configurator
import os.path

from scipy import sparse
from sklearn import datasets


class Controller:
    """Read and parse the config file, init a FeatureManager,
     and init a classifier manager. Handle output. """

    def __init__(self, configFiles=None):
        self.configFiles = configFiles
        self.configurators = []

        # classification parameters are fixed across Multilingual runs
        self.inputClasses = ""
        self.cv_folds = 1
        self.classifiersList = []
        self.threadsCount = 1
        self.featInput = ""
        self.featOutput = ""
        self.modelInput = ""
        self.modelOutput = ""
        self.featOutFormat = ""
        self.classifReport = ""
        self.train_size = 0
        self.val_size = 0
        self.random_state = None

        # array format of dataset and labels for classifying
        self.numSentences = 0
        self.extractedFeats = []
        self.unscaledFeats = []
        self.classesList = []

    def parseMergeConfigs(self):

        allFeats = []
        for config in self.configurators:
            allFeats.append(config.featureIDs)

            # Policy is to be the greatest
            if config.cv_folds > self.cv_folds:
                self.cv_folds = config.cv_folds
            if config.threadsCount > self.threadsCount:
                self.threadsCount = config.threadsCount

            # Only once or last instance
            # Todo: report conflicts
            if config.inputClasses:
                self.inputClasses = config.inputClasses
            if config.featOutput:
                self.featOutput = config.featOutput
            if config.featOutFormat:
                self.featOutFormat = config.featOutFormat
            if config.classifReport:
                self.classifReport = config.classifReport
            if config.featInput:
                self.featInput = config.featInput
            if config.modelInput:
                self.modelInput = config.modelInput
            if config.modelOutput:
                self.modelOutput = config.modelOutput
            if config.train_size:
                self.train_size = config.train_size
            if config.val_size:
                self.val_size = config.val_size
            if config.random_state is not None:
                self.random_state = config.random_state

            # Classifiers in different configs are merged
            if config.classifiersList:
                self.classifiersList.extend(config.classifiersList)

        self.classifiersList = list(set(self.classifiersList))

        return allFeats

    def loadConfig(self):
        """Read the config file(s), extract the featureIDs and
        their argument strings.
        """
        statusOK = 1

        # Extract featureID and feature Argument string
        for configFile in self.configFiles:
            with open(configFile) as config:
                # Parse the config file
                configurator = Configurator()
                statusOK = configurator.parseConfig(config)
                self.configurators.append(configurator)

                if not configurator.inputFile and statusOK:
                    print("Error, Missing input files.")
                    exit()

        mergedFeats = self.parseMergeConfigs()

        if not self.inputClasses and (self.classifiersList or self.featOutput):
            print("Error, Missing input files.")
            exit()

        return statusOK, mergedFeats, self.classifiersList

    def classesSentsMismatch(self, inputFile):
        if self.inputClasses:
            # Extract the classed IDs from the given classes file and Check for
            # Length equality with the sentences.
            prep_serv = Preprocess_Services()
            if not self.classesList:
                self.classesList = prep_serv.preprocessClassID(self.inputClasses)
            sentLen = len(prep_serv.preprocessBySentence(inputFile))
            classesLen = len(self.classesList)
            self.numSentences = sentLen
            if sentLen != classesLen:
                return True
        return False

    def manageFeatures(self):
        """Init and call a feature manager. """

        for configurator in self.configurators:
            if self.inputClasses and self.classesSentsMismatch(configurator.inputFile):
                print("Classes and Sentences length differ. Quiting. ")
                return 0


        if self.featInput: # read features from file
            self.extractedFeats = datasets.load_svmlight_file(self.featInput)[0]
            self.scaleFeatures()
            print(self.extractedFeats.shape)
        else:
            extractedFeats = []
            for configurator in self.configurators:
                manageFeatures = featman.Feature_manager(self.numSentences, configurator)
                validFeats = manageFeatures.checkFeatValidity()
                if validFeats:
                    # Continue to call features
                    extractedFeats.append(manageFeatures.callExtractors())
                else:
                    # terminate
                    print("Requested Feature ID not available.")
                    return 0
            self.extractedFeats = featman.mergeFeats(extractedFeats)
            self.unscaledFeats = self.extractedFeats.copy()
            self.scaleFeatures()
            self.outputFeatures()

        print("Feature Extraction Done. ")

        return 1

    def scaleFeatures(self):
        from sklearn import preprocessing as skpreprocess
        scaler = skpreprocess.MaxAbsScaler(copy=False)
        self.extractedFeats = scaler.fit_transform(self.extractedFeats)
        # TODO: make this a feature!



        #scaler1 = skpreprocess.MaxAbsScaler(copy=False)
        #self.extractedFeats = scaler1.fit_transform(self.extractedFeats)

        # scaler = skpreprocess.StandardScaler(with_std=False)
        #scaler = skpreprocess.MinMaxScaler(copy=False)
        # Standard Scaler (and MinMax) cannot deal with sparse matrices
        # We first unsparcify:
        #print("Shape of sparse matrix:", self.extractedFeats.shape)
        #self.extractedFeats = self.extractedFeats.toarray()
        #print("Shape of unsparcified matrix:", self.extractedFeats.shape)
        # Scale
        #self.extractedFeats = scaler.fit_transform(self.extractedFeats)
        # And sparcify again:
        #self.extractedFeats = sparse.lil_matrix(self.extractedFeats)
        #print("Shape of newly sparcified matrix:", self.extractedFeats.shape)


    def outputFeatures(self):
        """Output features if requested."""
        # TODO: currently writing unscaled features into file.
        # Maybe should make it a config option

        if self.featOutput:
            #formatter = format.Format(self.extractedFeats, self.classesList)
            formatter = format.Format(self.unscaledFeats, self.classesList)
            # if format is not set in config, will use a default libsvm output.
            formatter.outFormat(self.featOutput, self.featOutFormat)
        else:
            print("Feature output was not specified.")

    def classifyFeats(self):
        """Instantiate a classifier Manager then run it. """

        if self.inputClasses and self.classifiersList:
            # Classify if the parameters needed are specified
            classifying = classifier_manager.Classifier_manager(
                          self.classifiersList, self.extractedFeats, self.classesList,
                            self.threadsCount, self.cv_folds, self.modelInput,
                              self.modelOutput, self.train_size, self.val_size,
                                self.random_state)

            validClassifiers = classifying.checkValidClassifier()

            if validClassifiers:
                # Continue to call classifiers
                reportOfClassif = classifying.callClassifiers()
                print(reportOfClassif)
                # Write output if file specified
                if self.classifReport:
                    with open(self.classifReport, 'w') as classifOut:
                        classifOut.write(reportOfClassif)
                return 0
            else:
                # terminate
                print("Requested Classifier not available.")
                return -1
        else:
            print("Classifier parameters not specified.")
        return 1

