from dbfread import DBF
from scipy.spatial import distance
from hypergraphtk.core.hyper_graph_jit import *
import numpy as np
import googlemaps

class dbfile:

    def __init__(self, filename):
        """
        :param filename: requires full path name and file extension
        """
        self.filename =  filename
        self.output = []
        self.headers = []
        self.gmaps = googlemaps.Client(key='AIzaSyBWEqdC3EjiDzljsSvJi3v6wUWNKQZAl_g')


    def openfile(self):
        for record in DBF(self.filename):
            row = []
            #print(record)
            for i in record.items():
                if i[0] in self.headers:
                    pass
                else:
                    self.headers.append(i[0])
                row.append(i[1])
            self.output.append(row)
        self.output.insert(0, self.headers)
        return self.output


    def calc_elevation(self, lat, lon):
        """
        :param lat: float latitude
        :param lon: float longitude
        :return: list of lists
        """
        elevation = []
        self.headers.append('Elevation')
        lat_index = self.headers.index(lat)
        lon_index = self.headers.index(lon)

        tmp = []
        count = 0
        for i in self.output:
            if count == 0:
                pass
            else:
                latitude = float(i[lat_index])
                longitude = float(i[lon_index])
                tmp.append((latitude, longitude))
            count = count + 1

            if len(tmp) > 499: # API Rate Limit 2500 queries per day | 520 points per query
                geocode_result = self.gmaps.elevation(tmp)
                for k in geocode_result:
                    ele = k['elevation']
                    nlat = k['location']['lat']
                    nlon = k['location']['lng']
                    # convert to feet
                    ele = ele * 3.28084
                    loc = [(nlat, nlon), ele]
                    elevation.append(loc)
                tmp = []
            else:
                pass

        if len(tmp) > 0:
            geocode_result = self.gmaps.elevation(tmp)
            for l in geocode_result:
                ele = l['elevation']
                nlat = l['location']['lat']
                nlon = l['location']['lng']
                # convert to feet
                ele = ele * 3.28084
                loc = [(nlat, nlon), ele]
                elevation.append(loc)
        else:
            pass

        elevation_output = []
        count = 0
        for i in self.output:
            if count == 0:
                elevation_output.append(i)
            else:
                # print(new[count-1], elevation[count-1][0])
                i.append(elevation[count-1][1])
                elevation_output.append(i)
            count = count + 1

        self.output = elevation_output
        return self.output


    def construct_adjacency(self, lat, lon):
        """
        
        :param lat: float latitude
        :param lon: float longitude
        :return: numpy incidence matrix
        """
        lat_index = self.headers.index(lat)
        lon_index = self.headers.index(lon)

        point_list = []
        count = 0
        for i in self.output:
            if count == 0:
                pass
            else:
                point_list.append((i[lat_index], i[lon_index]))
            count = count + 1

        d = distance.cdist(point_list, point_list, 'euclidean')

        normed = d * 100

        proximity = max(normed[0])/(max(normed[0])*7)
        incident = []
        for j in normed:
            k = np.piecewise(j, [j <= proximity, j > proximity], [1, 0])
            incident.append(k)
        incident = np.vstack(incident)
        return incident


    def computeImpactCost(self, incident, hfield, tfield, lfield, bfield, disturbance_points, bloss_fun, lloss_fun, threshold):
        """
        :param incident: takes a numpy incidence matrix
        :param hfield: hyperedge field | parcels/points
        :param tfield: traffic field | elevations
        :param lfield: land value field
        :param bfield: building value field
        :param disturbance_points: list of parcel IDs
        :param bloss_fun: float point value | percent of value lost (buildings) given an event
        :param lloss_fun: float point value | percent of value lost (land) given an event
        :param threshold: flood impact level in feet
        :return: 
        """
        # Need to begin with the hyperedges, this case parcels.
        # Each parcel id represents a simplex
        parcel_index = self.headers.index(hfield)
        elevation_index = self.headers.index(tfield)
        bvalue_index = self.headers.index(bfield)
        lvalue_index = self.headers.index(lfield)
        # Initialize empty sets
        # Edges contain parcel IDs
        parcels = []
        # Traffic values actually represent Elevations
        elevations = []
        # Dollar value of a parcel
        land_values = []
        # Dollar value of a building on a parcel
        building_values = []
        # Collect all parcel id
        # collect all land values for each parcel
        # collect all building values for each parcel
        cnt = 0
        for i in self.output:
            if cnt == 0:
                pass
            else:
                parcels.append(i[parcel_index])
                land_values.append(i[lvalue_index])
                building_values.append(i[bvalue_index])
                # Exclude those parcels where elevation is greater than the threshold (flood level)
                if i[elevation_index] > threshold:
                    elevations.append(0)
                else: # include the value of parcels that could be affected
                    elevations.append(i[elevation_index])
            cnt = cnt + 1
        # Initialize hypergraph using original implementation or the jit implementation
        hgraph = hypergraph_jit(parcels, parcels) #jit version
        # Compute the pattern over the incidence matrix representation
        new_matrix = computePattern(tuple(elevations), incident)
        # returns matrix of zeros and float elevation values.
        # values greater than zero means a 1 was present in the incidence matrix.
        print(new_matrix)
        # Run Q-ana the new matrix and threshold value
        Q = hgraph.computeSimpleQ(new_matrix, 0)
        # returns
        print(Q)
        # Iterate through disturbance points
        # check if disturbance points occur within a given Eq. Class
        impacted_points = []
        for i in disturbance_points:
            # Iterate through each set of complexes
            for j in Q[1]:
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
        building_start_value = []
        land_start_value = []

        impacted_total_value = []
        impacted_building_value = []
        impacted_land_value = []

        for i in range(len(parcels)):
            building_start_value.append(building_values[i])
            land_start_value.append(land_values[i])

        for i in range(len(parcels)):
            if parcels[i] in impacted_points:
                nval = (building_values[i] - (building_values[i] * bloss_fun)) + (land_values[i] - (land_values[i] * lloss_fun))
                bval = building_values[i] - (building_values[i] * bloss_fun)
                lval = land_values[i] - (land_values[i] * lloss_fun)

                impacted_total_value.append(nval)
                impacted_building_value.append(bval)
                impacted_land_value.append(lval)
            else:
                nval = building_values[i]
                lval = land_values[i]
                bval = building_values[i]

                impacted_total_value.append(nval)
                impacted_building_value.append(bval)
                impacted_land_value.append(lval)


        building_value_loss = sum(building_start_value) - sum(impacted_building_value)
        land_value_loss = sum(land_start_value) - sum(impacted_land_value)
        total_start = sum(building_start_value) + sum(land_start_value)
        total_after = sum(impacted_building_value) + sum(impacted_land_value)
        total_loss = building_value_loss + land_value_loss

        print("Pre-flood value 'Built-Structures + Land': ", total_start)
        print("Post-flood value: ", total_after)
        print("Total value lost: ", total_loss)
        print("Pre-flood building value: ", sum(building_start_value))
        print("Post-flood building value: ", sum(impacted_building_value))
        print("Building value lost: ", building_value_loss)
        print("Pre-flood land value: ", sum(land_start_value))
        print("Post-flood land value: ", sum(impacted_land_value))
        print("Land value lost: ", land_value_loss)

        self.headers.append('PREFLOODVAL')
        self.headers.append("POSTFLOODVAL")
        self.headers.append("PREFLBUILDVAL")
        self.headers.append("POSTFLBUILDVAL")
        self.headers.append("PREFLLANDVAL")
        self.headers.append("POSTFLLANDDVAL")

        print(self.output[0])

