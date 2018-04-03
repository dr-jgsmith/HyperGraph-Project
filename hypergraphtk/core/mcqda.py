from hypergraphtk.core.hyper_graph import *


class mcqda:

    '''
    The mcqda class provides and implementation of both MCQA I and II.
    Multi-criteria Q-analysis is a method for using q-analysis as a multi-dimensional
    decision optimization technique.
    
    Q-Analysis is a method for studying the connective structure of relations between two sets.    
    Experimental use in discourse analysis, vector weights as a type of graded mapping on a complex to 
    determine similarity.
    '''
    def __init__(self, hyperedges, vertex_set):
        # Initialize with a set of hyperedges and vertices
        self.hyperedges = hyperedges
        self.vertex_set = vertex_set
        # Sets should go through the hypergraph class
        self.data = hypergraph(hyperedges, vertex_set)
        self.weights = {}

    #Set the weights | Note these can be probability distributions, tf-idf
    def set_weights(self, vector_weights):
        self.weights = vector_weights
        return self.weights


    def normMatrix(self, matrix):
        self.normed = self.data.normPositive(matrix=matrix)
        return self.normed


    def computePSIN(self, psi_dict, psimax):
        psi = {}
        for i in self.hyperedges:
            psi[i] = 0
            for j in psi_dict:
                psi[i] = psi[i] + j[i]

        psin = {}
        for i in psi.items():
            psin[i[0]] = i[1] / psimax

        return psi, psin


    def computePCIN(self, pci_dict, pcimax):
        pci = {}
        for i in self.hyperedges:
            pci[i] = 0
            for j in pci_dict:
                pci[i] = pci[i] + j[i]

        pcin = {}
        for i in pci.items():
            pcin[i[0]] = i[1] / pcimax

        return pci, pcin


    def computePDIN(self, pdi_dict, pdimax):
        pdi = {}
        for i in self.hyperedges:
            pdi[i] = 0
            for j in pdi_dict:
                pdi[i] = pdi[i] + j[i]

        pdin = {}
        for i in pdi.items():
            pdin[i[0]] = i[1] / pdimax

        return pdi, pdin

    def computeRanking(self, psi, pci, pdi):
        ranking = {}
        for i in self.hyperedges:
            si = psi.get(i)
            ci = pci.get(i)
            di = pdi.get(i)

            ri = (1 - si) + (1 - ci) + (di)
            ranking[i] = ri
        return ranking


    def computeMcqda(self, cut_points):
        scores = []
        compscor = []
        discore = []
        psimax = 0
        pcimax = len(self.vertex_set)
        for i in cut_points:
            Q = self.data.computeQanalysis(self.normed, theta=i, norm_matrix=True, conjugate=False)
            incident = self.data.incident
            psii = self.data.computePSIi(incident, self.weights)
            pcii = self.data.computePCIi(Q[0])

            Qpdi = self.data.computeQcompliment()

            pdii = self.data.computePDIi(Qpdi[1])
            scores.append(psii)
            compscor.append(pcii)
            discore.append(pdii)
            psimax = psimax + sum([i[1] for i in self.weights.items()])

        psi = self.computePSIN(scores, psimax)
        pci = self.computePCIN(compscor, pcimax)
        pdi = self.computePDIN(discore, pcimax)
        ranking = self.computeRanking(psi[1], pci[1], pdi[1])

        return psi, pci, pdi, ranking







