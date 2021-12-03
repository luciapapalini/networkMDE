/*
Module introduced for fast MDE calculation.

Network must be given in normalized mode:
  -> node labels start from 0
  -> no holes in labels
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cutils.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct sparserow
{
    long i;
    long j;
    double d;
} SparseRow;

typedef struct node
{
    long n;   // Node label
    double value;       // Value of a general scalar quantity
    long childs_number;
    long * childs;
    double * distances;
    double * position; // Position in the embedding
} Node;

// Link structure is useless at the moment,
// may bind it to Linkin python later
typedef struct link
{
    Node * node1;
    Node * node2;
    double distance;
} Link;

typedef struct graph
{
    long N_nodes;
    long N_links;
    Node * nodes;
    int embedding_dimension;
} Graph;

// Global variables remain the same call after call
Graph G;

// Link nodes in the Graph
void link_nodes(Node * node, long child_index, double distance){

    // Adds label of child to child array
    long * new_childs = (long *) malloc(sizeof(long)*((*node).childs_number + 1));
    for (long k = 0; k < node -> childs_number; k++)
    {
        if (node -> childs_number != 0)
        {
            new_childs[k] = node->childs[k];
        }
    }
    new_childs[node -> childs_number] = child_index;
    (*node).childs = new_childs;

    // Adds distances to distances array
    double * new_distances = (double *) malloc(sizeof(double)*((*node).childs_number + 1));
    for (long k = 0; k < node -> childs_number; k++)
    {
        new_distances[k] = node->distances[k];
    }
    new_distances[node -> childs_number] = distance;
    node -> distances = new_distances;

    node -> childs_number = (node -> childs_number) + 1;
    return ;
}

Graph to_Net(SparseRow * SM, double * values, long N_elements, long N_links){

    Graph g;
    g.nodes = (Node *) malloc(sizeof(Node)*N_elements);

    // Values assignment
    for (long k = 0; k < N_elements; k++)
    {
        g.nodes[k].n = k;
        g.nodes[k].value = values[k];
        g.nodes[k].childs_number = 0;
    }

    // Linking
    for (long k = 0; k < N_links; k++)
    {
        link_nodes(&(g.nodes[SM[k].i]), SM[k].j, SM[k].d);
        link_nodes(&(g.nodes[SM[k].j]), SM[k].i, SM[k].d);
    }

    g.N_nodes = N_elements;
    g.N_links = N_links;

    return g;
}

double * PyList_to_double(PyObject * Pylist, long N_elements){

    double * dlist = (double *) malloc(sizeof(double)*N_elements);
    for (long i = 0; i < N_elements; i++)
    {
        dlist[i] = PyFloat_AS_DOUBLE(PyList_GetItem(Pylist, i));
    }
    return dlist;
}

SparseRow * PyList_to_SM(PyObject * list, long N_links){

    SparseRow * SM = (SparseRow *) malloc(sizeof(SparseRow)*N_links);
    for (long k = 0; k < N_links;k++)
    {
        PyObject * row = PyList_GetItem(list,k);
        if (!PyList_Check(row))
        {
            printf("Invalid sparse list (row %li)\n", k);
            return NULL;
    }
    SM[k].i = PyLong_AsLong(PyList_GetItem(row,0));
    SM[k].j = PyLong_AsLong(PyList_GetItem(row,1));
    SM[k].d = PyFloat_AsDouble(PyList_GetItem(row,2));
    }
    return SM;
}

void random_init(){

    srand(time(0));
    for (int n = 0; n < G.N_nodes; n++)
    {
        G.nodes[n].position = (double *) malloc(sizeof(double)*G.embedding_dimension);
        for (int d = 0; d < G.embedding_dimension; d++)
        {
            G.nodes[n].position[d] = ((float) rand())/((float) RAND_MAX);
        }
    }

}

PyObject * init_network(PyObject * self, PyObject * args){

    // The PyObjs for the two lists
    PyObject * Psparse = NULL;
    PyObject * Pvalues = NULL;
    int embedding_dim = 0;

    // The C sparse matrix and the values array
    SparseRow * SM = NULL;
    double * values = NULL;

    // Take the args and divide them in two pyobjects
    if (!PyArg_ParseTuple(args,"OOi",&Psparse, &Pvalues, &embedding_dim)) Py_RETURN_NONE;
    printf("cnets - Parsing stage passed\n");

    long N_elements = (long) PyList_Size(Pvalues);
    long N_links = (long) PyList_Size(Psparse);
    printf("cnets - N_element & N_links stage passed\n");

    // Convert each element of the lists into a valid element of the C-object
    SM = PyList_to_SM(Psparse, N_links);
    values = PyList_to_double(Pvalues, N_elements);
    printf("cnets - Conversion stage passed\n");

    G = to_Net(SM, values, N_elements, N_links);
    G.embedding_dimension = embedding_dim;
    printf("cnets - Network generation stage passed\n");

    // Initializes the position randomly
    printf("cnets - Embedding_dimension = %d\n", G.embedding_dimension);
    random_init();
    printf("cnets - random_init() passed\n");
    Py_RETURN_NONE;
}

double distance(double * pos1, double * pos2, int dim){

    double dist = 0.;
    for (int i = 0; i < dim; i++)
    {
        dist += pow(pos1[i] - pos2[i], 2);
    }
    return sqrt(dist);
}

PyObject * MDE(PyObject * self, PyObject * args){

    double eps = 0.;
    int number_of_steps = 0;

    if (!PyArg_ParseTuple(args, "di", &eps, &number_of_steps))
    {
        Py_RETURN_NONE;
    }
    printf("cnets - starting MDE with eps = %lf, Nsteps = %d\n",eps, number_of_steps);

    double actual_distance = 0., factor;
    int child_index;
    for (int i = 0; i < number_of_steps; i++)
    {   
        progress_bar(((float)i)/( (float) number_of_steps) , 60);
        for (int current_node = 0; current_node < G.N_nodes; current_node++)
        {
            for (int current_child = 0; current_child < G.nodes[current_node].childs_number; current_child++ )
            {
                child_index = G.nodes[current_node].childs[current_child];
                actual_distance = distance(G.nodes[current_node].position, G.nodes[child_index].position, G.embedding_dimension);                
                factor = eps*(1.- G.nodes[current_node].distances[current_child]/actual_distance)/G.nodes[current_node].childs_number;
                
                for (int d = 0; d < G.embedding_dimension; d++)
                {
                    G.nodes[current_node].position[d] += factor*(G.nodes[child_index].position[d] - G.nodes[current_node].position[d]) ;
                }
            }
        }
    }
    printf("\ncnets - MDE end\n");
    Py_RETURN_NONE;
}

PyObject * get_positions(PyObject * self, PyObject * args){
    PyObject * list = PyList_New(G.N_nodes);
    for (int n = 0; n < G.N_nodes; n++)
    {
        PyObject * single = PyList_New(G.embedding_dimension);
        for (int d = 0; d < G.embedding_dimension; d++)
        {
            PyList_SetItem(single, d, PyFloat_FromDouble(G.nodes[n].position[d]));
        }
        PyList_SetItem(list, n, single);
    }
    return list;
}

PyObject * get_distanceSM(PyObject * self, PyObject * args){

    PyObject * distanceSM = PyList_New(G.N_nodes*G.N_nodes);
    double d;
    int row_index = 0;
    Node node, another_node;

    for (int node_index = 0; node_index < G.N_nodes; node_index++)
    {   
        for (int another_node_index = 0; another_node_index < G.N_nodes; another_node_index++)
        {   
            PyObject * row = PyList_New(3);

            node = G.nodes[node_index];
            another_node = G.nodes[another_node_index];

            d = distance(node.position, another_node.position, G.embedding_dimension);
            PyList_SetItem(row, 0, PyLong_FromLong(node.n));
            PyList_SetItem(row, 1, PyLong_FromLong(another_node.n));
            PyList_SetItem(row, 2, PyFloat_FromDouble(d));

            PyList_SetItem(distanceSM, row_index, row);
            row_index ++;
        }
        
    }
    return distanceSM;
}

// Python link part - follow the API

// Methods table definition
static PyMethodDef cnetsMethods[] = {
    {"init_network", init_network, METH_VARARGS, "Initializes the network given a sparse list and a list of values"},
    {"MDE", MDE, METH_VARARGS, "Executes minumum distortion embedding routine"},
    {"get_positions", get_positions, METH_VARARGS, "Gives the computed positions of the network"},
    {"get_distanceSM", get_distanceSM, METH_VARARGS, "Returns the computed distance sparse matrix"},
    {NULL, NULL, 0, NULL}//Guardian of The Table
};

// Module definition
static struct PyModuleDef cnetsmodule = {
    PyModuleDef_HEAD_INIT,
    "cnets",
    "Module for fast network computing",
    -1,
    cnetsMethods
};

// Initialization function for the module
PyMODINIT_FUNC PyInit_cnets(void) {
    return PyModule_Create(&cnetsmodule);
}
