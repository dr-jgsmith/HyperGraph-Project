import dataset
from stuf import stuf


class hyperdb:
    '''
    Hyperdb stores hyperedges, vertices and vertex weights or values
    
    Methods can be used to store relations between edges and vertices, and 
    for retrieving these relations for matrix construction and hypergraph computing
    based on Dowker complexes and Q-Analysis.
    
    Basic usage:
        db_name = "sqlite:///test.db"
        db = hyperdb(db=db_name)
        db.add_hyperedge('test', 'test2', 'stuff', '1')
        db.get_all_hyperedges()
    '''

    def __init__(self, db='default'):
        # Create a connection to a SQL DB.
        # Initialize with 'stuf' enabled to retrieve dictionary like data
        # You can turn this off.
        # For more information on dataset, refer to their website
        # https://dataset.readthedocs.io/en/latest/
        
        # Establish connection
        if db is 'default':
            self.db = dataset.connect("sqlite:///hypergraph.db", row_type=stuf)
        else:
            self.db = dataset.connect(db, row_type=stuf)

    def add_hyperedge(self, title, hyperedge, vertex, value):
        # Add entry to db
        if len(vertex) > 0:
            table = self.db['hypergraph']
            table.insert(dict(title=title, hyperedge=hyperedge, vertex=vertex, value=value))
            # Add hyperedge to hyperedge table
            self.update_edge_set(hyperedge)
            # Add vertex to vertex table
            self.update_vertex_set(vertex)
            return
        else:
            pass

    def update_edge_set(self, hyperedge):
        # Check if edge exists in hyperedges table
        edge_table = self.db['hyperedges']
        # Get all entries associated with a given hyperedge.
        edge = edge_table.find_one(hyperedge=hyperedge)
        # if edge exists, skip, otherwise add the edge to the Hyperedges Table
        # The Hyperedge set is used in constructing sets of simplicies for Q-Analysis.
        if edge:
            pass
        else:
            edge_table.insert(dict(hyperedge=hyperedge))
        return

    def update_vertex_set(self, vertex):
        # Check if vertex exists in the vertices table
        vertex_table = self.db['vertices']
        vert = vertex_table.find_one(vertex=vertex)
        # If the vertex exists in the table, pass. Otherwise update
        if vert:
            pass
        else:
            # Add new vertex
            vertex_table.insert(dict(vertex=vertex))
        return

    def get_all_hyperedges(self):
        # Retrieve all rows in the hypergraph table
        table = self.db['hypergraph'].all()
        return list(table)

    def get_hyperedge_set(self):
        # retrieve only a set of edges
        edge_table = self.db['hyperedges'].all()
        edge_set = [i.get('hyperedge') for i in list(edge_table)]
        return edge_set

    def get_vertex_set(self):
        # retrieve only a set of vertices
        vertex_table = self.db['vertices'].all()
        vertex_set = [i.get('vertex') for i in list(vertex_table)]
        return vertex_set

    def get_edge_vertices(self, hyperedge):
        # Retrieve all vertices for a given hyperedge
        table = self.db['hypergraph']
        results = table.find(hyperedge=hyperedge)
        return results

    def get_matrix(self):
        # Construct matrix repreentation of the data
        # The matrix construction is required for performing the Q-analysis
        vertex_set = self.get_vertex_set()
        hyperedge_set = self.get_hyperedge_set()
        matrix = []
        for i in hyperedge_set:
            row = [0] * len(vertex_set)
            vertices = self.get_edge_vertices(i)
            for j in vertices:
                loc = vertex_set.index(j['vertex'])
                row[loc] = j['value']
            matrix.append(row)
        return matrix

