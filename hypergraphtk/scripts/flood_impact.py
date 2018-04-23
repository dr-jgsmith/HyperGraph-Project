from hypergraphtk.storage.processdbf import *
from scipy.spatial import distance
from hypergraphtk.core.hyper_graph_jit import *
import numpy as np
import googlemaps


class flood_model:

    def __init__(self, filename):
        """
        :param filename: requires full path name and file extension
        """
        self.file = processdbf(filename)
        self.file.openfile()
        self.gmaps = googlemaps.Client(key='XXXXXXXXXXXXXXXXXXX')


    def calc_elevation(self, latfield, lonfield, newfield):
        """
        :param lat: float latitude
        :param lon: float longitude
        :return: list of lists
        """
        elevation = []

        lat_index = self.file.get_column(latfield)
        lon_index = self.file.get_column(lonfield)

        tmp = []
        for i in range(len(lat_index[1])):
            tmp.append((float(lat_index[1][i]), float(lon_index[1][i])))

            if len(tmp) > 499:  # API Rate Limit 2500 queries per day | 520 points per query
                geocode_result = self.gmaps.elevation(tmp)
                for k in geocode_result:
                    ele = k['elevation']
                    # convert to feet
                    ele = ele * 3.28084
                    elevation.append(ele)
                tmp = []
            else:
                pass


        if len(tmp) > 0:
            geocode_result = self.gmaps.elevation(tmp)
            for l in geocode_result:
                ele = l['elevation']
                # convert to feet
                ele = ele * 3.28084
                elevation.append(ele)
        else:
            pass

        self.file.add_column(newfield, elevation)
        return self.file.output


    def construct_adjacency(self, latfield, lonfield):
        """

        :param lat: float latitude
        :param lon: float longitude
        :return: numpy incidence matrix
        """
        lat_index = self.file.get_column(latfield)
        lon_index = self.file.get_column(lonfield)

        point_list = []
        for i in range(len(lat_index[1])):
            point_list.append((lat_index[1][i], lon_index[1][i]))

        d = distance.cdist(point_list, point_list, 'euclidean')

        normed = d * 100

        proximity = max(normed[0]) / (max(normed[0]) * 7)
        incident = []
        for j in normed:
            k = np.piecewise(j, [j <= proximity, j > proximity], [1, 0])
            incident.append(k)
        incident = np.vstack(incident)
        return incident


    def computeImpactZones(self, incident, parcel_field, elevation_field, threshold):
        """
        :param incident: 
        :param parcel_field: 
        :param elevation_field: 
        :param threshold: 
        :return: 
        """
        # Need to begin with the hyperedges, this case parcels.
        # Each parcel id represents a simplex
        parcels = self.file.get_column(parcel_field)
        elevation_index = self.file.get_column(elevation_field)
        # Initialize empty sets
        elevations = []
        for i in elevation_index[1]:
            if i > threshold:
                elevations.append(0)
            else:  # include the value of parcels that could be affected
                elevations.append(i)
        # Initialize hypergraph using original implementation or the jit implementation
        hgraph = hypergraph_jit(parcels[1], parcels[1])  # jit version
        # Compute the pattern over the incidence matrix representation
        new_matrix = computePattern(tuple(elevations), incident)
        # returns matrix of zeros and float elevation values.
        # values greater than zero means a 1 was present in the incidence matrix.
        print(new_matrix)
        # Run Q-ana the new matrix and threshold value
        Q = hgraph.computeSimpleQ(new_matrix, 0)
        # returns
        return Q


    def computeImpactCost(self, zones, parcel_field, bfield, lfield, impact_point, bloss_fun, lloss_fun):
        """
        :param zones: list of connected parcel sets
        :param parcel_field: field for parcel ids
        :param bfield: building value field name
        :param lfield: land value field name
        :param impact_point: a list of initial disturbance points
        :param bloss_fun: 
        :param lloss_fun: 
        :return: 
        """
        parcels = self.file.get_column(parcel_field)
        building_values = self.file.get_column(bfield)
        land_values = self.file.get_column(lfield)

        impacted_points = []
        for i in impact_point:
            # Iterate through each set of complexes
            for j in zones:
                # if a disturbance point is present within a complex
                if i in j:
                    # collect all parcels within the effected complex
                    [impacted_points.append(k) for k in j]
                else:
                    pass
        # compute impact based on number of impacted parcels in q-component | connected zone
        # connected zone was computed as the sharing of proximity and elevation among parcels.
        # for parcels to be affected by the flood they need to be in close proximity and of similar elevation
        # note variation of topography can produce significant variation, thus computing with a topographic image would be ore precise.

        impacted_total_value = []
        impacted_building_value = []
        impacted_land_value = []

        for i in range(len(parcels[1])):
            if parcels[1][i] in impacted_points:
                nval = (building_values[1][i] - (building_values[1][i] * bloss_fun)) + (
                land_values[1][i] - (land_values[1][i] * lloss_fun))
                bval = building_values[1][i] - (building_values[1][i] * bloss_fun)
                lval = land_values[1][i] - (land_values[1][i] * lloss_fun)

                impacted_total_value.append(nval)
                impacted_building_value.append(bval)
                impacted_land_value.append(lval)
            else:
                nval = building_values[1][i]
                lval = land_values[1][i]
                bval = building_values[1][i]

                impacted_total_value.append(nval)
                impacted_building_value.append(bval)
                impacted_land_value.append(lval)

        building_value_loss = sum(building_values[1]) - sum(impacted_building_value)
        land_value_loss = sum(land_values[1]) - sum(impacted_land_value)
        total_start = sum(building_values[1]) + sum(land_values[1])
        total_after = sum(impacted_building_value) + sum(impacted_land_value)
        total_loss = building_value_loss + land_value_loss

        print("Pre-flood value 'Built-Structures + Land': ", total_start)
        print("Post-flood value: ", total_after)
        print("Total value lost: ", total_loss)
        print("Pre-flood building value: ", sum(building_values[1]))
        print("Post-flood building value: ", sum(impacted_building_value))
        print("Building value lost: ", building_value_loss)
        print("Pre-flood land value: ", sum(land_values[1]))
        print("Post-flood land value: ", sum(impacted_land_value))
        print("Land value lost: ", land_value_loss)

