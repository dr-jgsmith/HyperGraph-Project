from hypergraphtk.scripts.flood_impact import *


"""
This begins the example use case of value_impact_analysis

"""
# Begin with dBase file commonly associated with attribute data for annotating layers in a GIS
file = "C:\\Users\\justi\\Desktop\\Ab_Hoq_Cosi_Parcels_Lat_Long_Points\\Ab_Hoq_Cosi_Parcels_Lat_Long_Points.dbf"
# Open the file and intitialize the flood model class
db = flood_model(file)

# Define the column or id names
lat = 'Lat'
lon = 'Long'
bfield = 'BLDGVALUE'
lfield = 'LANDVALUE'
efield = 'PARCELATT'

# Define a new field, Elevation
tfield = 'Elevation'
# Define parameters
# The threshold parameter relates to flood water level in feet
threshold = 12.
# bloss_fun is the percent of value lost on building that has been effected by flood waters.
# This number can be modified. Or could be random within a range based on available data.
bloss_fun = 0.02
# lloss_fun is the percent of value lost on actual land.
# In a dynamical system, this might be dependent on frequency of events
lloss_fun = 0.001
# define a set of start points, these are parcels that would likely see the first impacts
# this allows for points to be defined close to bodies of water as well as points where infrastructure fails to accommodate flow.
disturbance_points = ['029901602600']
# calcuate an elevation level for each parcel using Google Maps API
d = db.calc_elevation(lat, lon, tfield)
print(d)
# Elevation is not a enough to determine the risk of a parcel, parcels need to be within in relative proximity.
# Topographic variation can create barriers to flooding, as well as the presence of mitigation structures.
# These features need a way to be defined and considered in an analysis.
# Here we compute proximity and define a type of threshold for connecting parcels by adjacency.
a = db.construct_adjacency(lat, lon)
print(a)
# Given the proximity and elevations we can compute a baseline of at risk parcel "zones"
# Zones are defined as a parcels with similar risk due to connectivity of proximity and similar elevations.
x = db.computeImpactZones(a, efield, tfield, threshold)
print(x)
# Generate an impact value for at risk properties in effected zones.
# Using the zones generated from the previous method, we can compute impact potential.
# The impact potential is determined by a set of fields and parameters.
# A list of disturbance points are defined, these are the points where an event first occurs
# then we need to pass a value for computing percent of value lost for both building and land as they are different.
db.computeImpactCost(x[1], efield, bfield, lfield, disturbance_points, bloss_fun, lloss_fun)
# A more dynamic approach would modify these for recurring events.
# To be continued...
