import os.path


class Configurator:
    """Read and parse the config file and return a config object """

    def __init__(self):
        self.featureIDs = []
        self.featargs = []
        self.inputClasses = []
        self.classifiersList = []
        self.inputFile = ""
        self.classifReport = ""
        self.modelInput = ""
        self.modelOutput = ""
        self.corpusLM = ""
        self.featInput = ""
        self.featOutput = ""
        self.featOutFormat = ""
        self.threadsCount = 1
        self.language = 'eng'
        self.srilmBinPath = ""
        self.kenlmBinPath = ""
        self.cv_folds = 1
        self.train_size = 0
        self.val_size = 0
        self.random_state = None # for the classifier

    def parseOutputLine(self, line):
        status = 1
        startInp = line.index(':')
        outputLine = line[startInp + 1:]
        outputLine = outputLine.strip().split()
        if line.startswith("output classifier") and not self.classifReport:
            self.classifReport = outputLine[0]
        elif line.startswith("output features") and not self.featOutput:
            if len(outputLine) == 2:
                self.featOutput = outputLine[0]
                self.featOutFormat = outputLine[1]
            elif len(outputLine) == 1:
                self.featOutput = outputLine[0]
            else:
                status = 0
                print("Incorrect number of output params, should be exactly 2")
        else:
            print("Unsupported output type")
            status = 0

        return status

    def parseConfig(self, configFile):
        """Parse the config file lines.      """
        statusOK = 1

        for configLine in configFile:
            configLine = configLine.strip()
            if not statusOK:
                break
            if len(configLine) < 1:
                # Line is empty
                continue
            elif configLine[0] is '#':
                # Line is comment
                continue
            elif configLine.startswith("input file"):
                # Extract input file
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                self.inputFile = configLine[0]
                print("Input file: ")
                print(self.inputFile)
            elif configLine.startswith("input classes"):
                # Extract input classes file
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                self.inputClasses = configLine[0]
                #print("Input classes: ")
                #print(self.inputClasses)
            elif configLine.startswith("output"):
                statusOK = self.parseOutputLine(configLine)
            elif configLine.startswith("classif"):
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                self.classifiersList = configLine
            elif configLine.startswith("training corpus"):
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                self.corpusLM = configLine[0]
            elif configLine.startswith("SRILM") or configLine.startswith("srilm"):
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip()
                self.srilmBinPath = configLine
                if not os.path.isdir(self.srilmBinPath):
                    statusOK = 0
                    print("Invalid SRILM binaries path.")
                else:
                    self.srilmBinPath = os.path.join(self.srilmBinPath, '')
            elif configLine.startswith("kenlm"):
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip()
                self.kenlmBinPath = configLine
                if not os.path.isdir(self.kenlmBinPath):
                    statusOK = 0
                    print("Invalid KenLm binaries path.")
                else:
                    self.kenlmBinPath = os.path.join(self.kenlmBinPath, '')
            elif configLine.startswith("operating language"):
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                self.language = configLine
                #print(self.language)
            elif configLine.startswith("thread"):
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                if configLine[0].isdigit():
                    threads = int(configLine[0])
                    if threads > 0:
                        #handle single thread case
                        self.threadsCount = threads if threads < 3 else threads-1
                    else:
                        statusOK = 0
                        print("Number of threads is not a positive integer.")
                    #print(self.threadsCount)
                else:
                    statusOK = 0
                    print("Number of threads is not a positive integer.")
            elif configLine.startswith("fold"):
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                if configLine[0].isdigit():
                    folds = int(configLine[0])
                    if folds > 0:
                        self.cv_folds = folds

                    else:
                        statusOK = 0
                        print("Number of folds is not a positive integer.")
                else:
                    statusOK = 0
                    print("Number of folds is not a positive integer.")
            elif configLine.startswith("input features"): # Load extracted feature matrix from file
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                self.featInput = configLine[0]
            elif configLine.startswith("load model"): # Load trained model from file
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                self.modelInput = configLine[0]
            elif configLine.startswith("save model"): # Load trained model from file
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                self.modelOutput = configLine[0]
            elif configLine.startswith("train size"):
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                if configLine[0].isdigit():
                    train_size = int(configLine[0])
                    if train_size > 0:
                        self.train_size = train_size
                    else:
                        statusOK = 0
                        print("Train size is not a positive integer.")
                else:
                    statusOK = 0
                    print("Train size is not a positive integer.")
            elif configLine.startswith("val size"):
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                if configLine[0].isdigit():
                    val_size = int(configLine[0])
                    if val_size > 0:
                        self.val_size = val_size
                    else:
                        statusOK = 0
                        print("Val size is not a positive integer.")
                else:
                    statusOK = 0
                    print("Val size is not a positive integer.")
            elif configLine.startswith("random state"):
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                if configLine[0].isdigit():
                    self.random_state = int(configLine[0])
                else:
                    statusOK = 0
                    print("Random state is not an integer.")
            else:
                params = str(configLine).split(' ', 1)
                if len(params) == 2 or len(params) == 1:
                    if params[0].isdigit():
                        self.featureIDs.append(int(params[0]))
                        if len(params) == 2:
                            self.featargs.append(params[1])
                        else:
                            self.featargs.append([])
                    else:
                        statusOK = 0
                        print("Feature ID is not a Number")
                else:
                    # Incorrect number/value of params
                    statusOK = 0
                    print("Incorrect number of params, max 2 parameters.")

        return statusOK

