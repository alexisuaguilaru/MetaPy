from DifferentialEvolution import DifferentialEvolution

class DifferentialEvolution_Clustering(DifferentialEvolution):
    def __init__(self, objectiveFunction, initializeIndividual):
        super().__init__(objectiveFunction, initializeIndividual)