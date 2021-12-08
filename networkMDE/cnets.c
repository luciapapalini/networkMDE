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
int RAND_INIT = 0;

// Link nodes in the Graph
void link_nodes(Node * node, long child_index, double distance){

    // Adds label of child to child array
    // printf("cnets - link_nodes - realloc childs (old = %p)...", node -> childs); fflush(stdout);
    node -> childs = (long *) realloc(node -> childs, sizeof(long)*( (node -> childs_number) + 1));
    if (node -> childs == NULL)
    {
        printf("!!! cannot allocate memory !!!\n");
        exit(-1);
    }
    // printf("assign new child..."); fflush(stdout);
    (node -> childs)[node -> childs_number] = child_index;
    // printf("Done.\n");

    // Adds distances to distances array
    // printf("cnets - link_nodes - realloc distances..."); fflush(stdout);
    node -> distances = (double *) realloc(node -> distances, sizeof(double)*((*node).childs_number + 1));
    if (node -> distances == NULL)
    {
        printf("!!! cannot allocate memory !!!\n");
        exit(-1);
    }
    // printf("assign new distance..."); fflush(stdout);
    (node -> distances)[node -> childs_number] = distance;
    // printf("Done.\n");

    node -> childs_number = (node -> childs_number) + 1;
    return ;
}

Graph to_Net(SparseRow * SM, double * values, long N_elements, long N_links){
    printf("cnets - to_Net - ALLOC\n");
    Graph g;
    g.nodes = (Node *) malloc(sizeof(Node)*N_elements);
    if (g.nodes == NULL)
    {
        printf("!!! cannot allocate memory for %ld nodes !!!\n", N_elements);
        exit(-1);
    }
    printf("cnets - to_Net - ASSIGN\n");
    // Values assignment
    for (long k = 0; k < N_elements; k++)
    {
        g.nodes[k].n = k;
        g.nodes[k].value = values[k];
        g.nodes[k].childs_number = 0;
        g.nodes[k].childs = (long *) malloc(sizeof(long)); // Since each node has at least one child
        g.nodes[k].distances = (double *) malloc(sizeof(double));
    }

    printf("cnets - to_Net - LINK\n");
    // Linking
    for (long k = 0; k < N_links; k++)
    {
        link_nodes(&(g.nodes[SM[k].i]), SM[k].j, SM[k].d);
        link_nodes(&(g.nodes[SM[k].j]), SM[k].i, SM[k].d);
    }
    printf("cnets - to_Net - NEl&NLi\n");
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

long child_local_index_by_child_name(long node_number, long child_name)
{
    for (int child_local_index = 0; child_local_index < G.nodes[node_number].childs_number; child_local_index++)
        {
            if (G.nodes[node_number].childs[child_local_index] == child_name)
            {
                return child_local_index;
            }
            if (child_local_index == G.nodes[node_number].childs_number - 1)
            {
                // printf("cnets - WARNING - Node %li has not child %li\n", node_number, child_name);
                return (long) -1;
            }
        }
}

void random_init(){
    if (RAND_INIT == 0)
    {
        srand(time(0));
    }
    else
    {
        srand(RAND_INIT);
    }

    for (int n = 0; n < G.N_nodes; n++)
    {
        G.nodes[n].position = (double *) malloc(sizeof(double)*G.embedding_dimension);
        for (int d = 0; d < G.embedding_dimension; d++)
        {
            G.nodes[n].position[d] = ((float) rand())/((float) RAND_MAX);
        }
    }

}

// Python enters here first 90% of the time
PyObject * init_network(PyObject * self, PyObject * args){

    // The PyObjs for the two lists
    PyObject * Psparse = NULL;
    PyObject * Pvalues = NULL;
    int embedding_dim = 0;

    // The C sparse matrix and the values array
    SparseRow * SM = NULL;
    double * values = NULL;

    // Take the args and divide them in two pyobjects
    printf("cnets - Parsing...");fflush(stdout);
    if (!PyArg_ParseTuple(args,"OOi",&Psparse, &Pvalues, &embedding_dim)) Py_RETURN_NONE;
    printf("\tDone.\n");

    long N_elements = (long) PyList_Size(Pvalues);
    long N_links = (long) PyList_Size(Psparse);
    if (N_links < 1 ||  N_elements < 2)
    {
        printf("cnets - invalid network G = (%ld, %ld)\n", N_elements, N_links);
        exit(2);
    }

    // Convert each element of the lists into a valid element of the C-object
    printf("cnets - Converting Py -> C..");fflush(stdout);
    SM = PyList_to_SM(Psparse, N_links);
    values = PyList_to_double(Pvalues, N_elements);
    printf("\tDone.\n");

    printf("cnets - Generating network...");fflush(stdout);
    G = to_Net(SM, values, N_elements, N_links);
    free(SM);
    free(values);
    G.embedding_dimension = embedding_dim;
    printf("\tDone.\n");

    // Initializes the position randomly
    printf("cnets - Random initialization in R%d...", G.embedding_dimension); fflush(stdout);
    random_init();
    printf("\tDone.\n");
    Py_RETURN_NONE;
}

double euclidean_distance(double * pos1, double * pos2, int dim){

    double dist = 0.;
    for (int i = 0; i < dim; i++)
    {
        dist += pow(pos1[i] - pos2[i], 2);
    }
    return sqrt(dist);
}

void move_away_from_random_not_child(long node, double eps){
    // Picks a guy at random until it is not a child
    long not_child;
    int draws = 0;
    do{
        not_child = (long)((G.N_nodes-1)*((float) rand()/RAND_MAX));
        draws++;
        if (draws > G.N_nodes){
            return;
        }
    }while(child_local_index_by_child_name(node, not_child) != (long)-1 || node == not_child);
    double dist = euclidean_distance(G.nodes[node].position, G.nodes[not_child].position, G.embedding_dimension);
    for (int d = 0; d < G.embedding_dimension; d++)
    {
        G.nodes[node].position[d] += eps/(dist*dist)*(G.nodes[node].position[d] - G.nodes[not_child].position[d]);
        if (G.nodes[node].position[d] != G.nodes[node].position[d]){
            printf("NAN detected\n");
            printf("%li- %li --> %lf\n", node, not_child, dist);
            exit(5);
        }
    }
    return;
}

PyObject * MDE(PyObject * self, PyObject * args){
    double eps = 0., neg_eps = 0.;
    int number_of_steps = 0;

    if (!PyArg_ParseTuple(args, "ddi", &eps, &neg_eps, &number_of_steps))
    {
        printf("cnets - ERROR parsing MDE args\n");
        Py_RETURN_NONE;
    }
    printf("cnets - starting MDE with eps = %lf, neg_eps = %lf, Nsteps = %d\n",eps, neg_eps, number_of_steps);

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
                actual_distance = euclidean_distance(G.nodes[current_node].position, G.nodes[child_index].position, G.embedding_dimension);                
                factor = eps*(1.- G.nodes[current_node].distances[current_child]/actual_distance)/G.nodes[current_node].childs_number;
                
                for (int d = 0; d < G.embedding_dimension; d++)
                {
                    G.nodes[current_node].position[d] += factor*(G.nodes[child_index].position[d] - G.nodes[current_node].position[d]) ;
                }
            }
            if (neg_eps != 0.){
                for (int mv_aw = 0; mv_aw < (int)(0.10*G.N_nodes); mv_aw++)
                {
                    move_away_from_random_not_child(current_node, neg_eps);
                }
            }
        }
    }
    printf("\ncnets - MDE end\n");
    Py_RETURN_NONE;
}

PyObject * get_positions(PyObject * self, PyObject * args){
    printf("cnets - Passing position back to python..."); fflush(stdout);
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
    printf("\tDone\n");
    return list;
}

PyObject * get_distanceSM(PyObject * self, PyObject * args)
{

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

            d = euclidean_distance(node.position, another_node.position, G.embedding_dimension);
            PyList_SetItem(row, 0, PyLong_FromLong(node.n));
            PyList_SetItem(row, 1, PyLong_FromLong(another_node.n));
            PyList_SetItem(row, 2, PyFloat_FromDouble(d));

            PyList_SetItem(distanceSM, row_index, row);
            row_index ++;
        }
        
    }
    return distanceSM;
}

PyObject * set_target(PyObject * self, PyObject * args)
{
    // printf("cnets - updating target distances\n");
    PyObject * PySM;
    long node1_number, node2_number;

    if(! PyArg_ParseTuple(args, "O", &PySM))
    {
        printf("cnets - set_target: paring failed\n");
    }
    SparseRow * SM = PyList_to_SM(PySM, G.N_links);
    for (int link = 0; link < G.N_links; link++)
    {
        node1_number = SM[link].i;
        node2_number = SM[link].j;
        // printf("cnet - set_target - (row %d of SM) --> (%li, %li) was (%lf), now is (%lf)\n", link, node1_number,node2_number, G.nodes[node1_number].distances[child_local_index_by_child_name(node1_number, node2_number)] , SM[link].d);
        
        G.nodes[node1_number].distances[child_local_index_by_child_name(node1_number, node2_number)] = SM[link].d;
        G.nodes[node2_number].distances[child_local_index_by_child_name(node2_number, node1_number)] = SM[link].d;

    }
    Py_RETURN_NONE;
}

PyObject * set_seed(PyObject * self, PyObject * args)
{
    int seed;
    if (!PyArg_ParseTuple(args, "i", &seed)){
        printf("cnets - parsing failed in set_seed()\n");
    }
    RAND_INIT = seed;
    Py_RETURN_NONE;
}

// Python link part - follow the API

// Methods table definition
static PyMethodDef cnetsMethods[] = {
    {"init_network", init_network, METH_VARARGS, "Initializes the network given a sparse list and a list of values"},
    {"MDE", MDE, METH_VARARGS, "Executes minumum distortion embedding routine"},
    {"get_positions", get_positions, METH_VARARGS, "Gives the computed positions of the network"},
    {"get_distanceSM", get_distanceSM, METH_VARARGS, "Returns the computed distance sparse matrix"},
    {"set_target", set_target, METH_VARARGS,"sets the target sparse matrix"},
    {"set_seed", set_seed, METH_VARARGS, "Set the seed for random numbers"},
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
