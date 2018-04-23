from hypergraphtk.core.hyper_graph import *

class mcqda:
    '''
    The mcqda class provides and implementation of both MCQA I and II.
    Multi-criteria Q-analysis is a method for using q-analysis as a multi-dimensional
    decision optimization technique.
    
    Q-Analysis is a method for studying the connective structure of relations between two finite sets.    
    Experimental use in discourse analysis, vector weights as a type of graded mapping on a complex to 
    determine similarity.
    
    Additional use and experimentation in as an impact analysis technique.
    '''
    def __init__(self, hyperedges, vertex_set):
        # Initialize with a set of hyperedges and vertices
        self.hyperedges = hyperedges
        self.vertex_set = vertex_set
        # Sets should go through the hypergraph class
        # initialize hypergraph object
        self.data = hypergraph(hyperedges, vertex_set)
        self.weights = {}


    # This is often a first step process -  generate a vector of
    # This is performed on all vertices in the data set
    def set_weights(self, vector_weights):
        # Set the weights | Note these can be probability distributions, tf-idf
        self.weights = vector_weights
        return self.weights


    # The next step is to normalize the vertices in the data set.
    # Results in a continuous value between 0 and 1.
    def normMatrix(self, matrix):
        # normalize the matrix | call the normPositive method on the hypergraph data object
        self.normed = self.data.normPositive(matrix)
        return self.normed


    # Compute the score for each cut-point.
    def computePSIi(self, new_matrix, weights):
        # This method can be used to collect a preference satisfaction index given a single slicing threshold/cut-point
        # To compute a complete preference satisfaction index requires this method to be called for each slice

        # retain the weights by index -
        # the sum of the weights represent the max value possible | PSIMAX
        vector_wght = [i[1] for i in weights.items()]
        count = 0
        psi = {}
        psin = {}
        # iterate through matrix
        for i in new_matrix:
            cnt = 0
            vals = []
            # multiply weights and incident vertices
            for j in i:
                val = vector_wght[cnt] * j
                vals.append(val)
                cnt = cnt + 1
            simplex = self.hyperedges[count]
            psi[simplex] = sum(vals)
            count = count + 1
        # compute the nromalized PSI
        for k in psi.items():
            psin[k[0]] = k[1] / sum(vector_wght)

        # return both the raw PSI and normalized PSI
        return psi, psin


    def computePCIi(self, qmatrix):
        # This method can be used to collect a preference comparison index given a single slicing threshold/cut-point
        # To compute a complete preference comparision index requires this method to be called for each slice
        # The repeated measure function is referenced in the mcqda.py file
        row = 0
        pci = {}
        pcin = {}
        # iterate through matrix
        for i in qmatrix:
            #get the simplex with the max value
            q_max = max(i)
            col = 0
            tmp = []
            # get the next the simplex with the next highest connected value
            # this is the diagnoal of the qmatrix
            for j in i:
                # if row index and column index are the same then skip
                # else append the value
                if row == col:
                    tmp.append(0)
                else:
                    tmp.append(j)
                col = col + 1
            # get the largest appended value
            q_star = max(tmp)
            # get the simplex id by row index value
            simplex = self.hyperedges[row]
            # compute pci value for simplex i
            pci[simplex] = q_max - q_star
            row = row + 1
        # Compute the normalized PCI
        for k in pci.items():
            pcin[k[0]] = k[1]  / (len(self.hyperedges) - 1)

        return pci, pcin


    def computePDIi(self, qmatrix):
        # This method can be used to collect a preference discordance index given a single slicing threshold/cut-point
        # To compute a complete preference discordance index requires this method to be called for each slice
        # The repeated measure function is referenced in the mcqda.py file
        row = 0
        pdi = {}
        pdin = {}
        for i in qmatrix:
            # get the simplex with the max value
            q_max = max(i)
            col = 0
            tmp = []
            # get the next the simplex with the next highest connected value
            # this is the diagnoal of the qmatrix
            for j in i:
                if row == col:
                    tmp.append(0)
                else:
                    tmp.append(j)
                col = col + 1
            # get the largest appended value
            q_star = max(tmp)
            # get the simplex id by row index value
            simplex = self.hyperedges[row]
            # compute pci value for simplex i
            pdi[simplex] = q_max - q_star
            row = row + 1
        # Compute the normalized PDI
        for k in pdi.items():
            pdin[k[0]] = k[1] / (len(self.hyperedges) - 1)

        return pdi, pdin


    def computeRankingI(self, psi, pci):
        # Compute composite ranking based on computed PSI, PCI scores
        ranking = {}
        # For each simplex in the hyperedges set
        for i in self.hyperedges:
            # get the psi value for i
            si = psi.get(i)
            # get the pci value for i
            ci = pci.get(i)

            # compute composite ranking for i
            ri = (1 - si) + (1 - ci)
            ranking[i] = ri
        return ranking


    # This method implements the MCQDA II algorithm that includes a preference discordance index
    def computeRankingII(self, psi, pci, pdi):
        # Compute composite ranking based on computed PSI, PCI, PDI scores
        ranking = {}
        # For each simplex in the hyperedges set
        for i in self.hyperedges:
            # get the psi value for i
            si = psi.get(i)
            # get the pci value for i
            ci = pci.get(i)
            # get the pdi value for i
            di = pdi.get(i)
            # compute composite ranking for i
            ri = (1 - si) + (1 - ci) + (di)
            ranking[i] = ri
        return ranking


    # This method takes a single cut-point for computing a preference ranking.
    def computeMcqda(self, cut_point):
        # The multi-criteria q-analysis method is based on computing a composite preference ranking
        # The composite scores are based on the Preference Satisfaction Index (PSI),
        # the Preference Comparision Index (PSI), and Preference Discordance Index (PDI)
        # Set empty values for computing PSI Scores, PCI Q Composite Score, and PDI Q-Discordance
        # perform the q-analysis for the threshold value i.
        Q = self.data.computeQanalysis(self.normed, cut_point, norm_matrix=True, conjugate=False)
        # get the incidence matrix
        incident = self.data.incident
        # The incident matrix is used to compute the PSI
        psii = self.computePSIi(incident, self.weights)
        # Use the Q-Dim Matrix to compute the PCI
        pcii = self.computePCIi(Q[0])
        # Compute the complement of the incidence matrix and re-run the Q-analysis
        Qpdi = self.data.computeQcompliment()
        # With the results of the Q-compliment we can compute the discordance index.
        pdii = self.computePDIi(Qpdi[1])
        # Rank the index as a composite score.
        # Note: this method includes the discordance index, but the PSI and PCI might be sufficient in many cases.
        rankingI = self.computeRankingI(psii[1], pcii[1])
        rankingII = self.computeRankingII(psii[1], pcii[1], pdii[1])

        return psii, pcii, pdii, rankingI, rankingII


    # This method will compute the MCQDA algorithm for each scale in a list of cut-points
    def process_mcqda_scales(self, cut_points):
        scales = {}
        for i in cut_points:
            scores = {}
            x = self.computeMcqda(i)
            print("print psi ", x[0][0])
            print("print psin ", x[0][1])
            scores['psi'] = x[0][0]
            scores['psin'] = x[0][1]

            print("print pci ", x[1][0])
            print("print pcin ", x[1][1])
            scores['pci'] = x[1][0]
            scores['pcin'] = x[1][1]

            print("print pdi ", x[2][0])
            print("print pdin ", x[2][1])
            scores['pdi'] = x[2][0]
            scores['pdin'] = x[2][1]

            print("print rank I ", x[3])
            print("print rank II", x[4])
            scores['rankI'] = x[3]
            scores['rankII'] = x[4]
            scales[str(i)] = scores

        return scales