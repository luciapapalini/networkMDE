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
    unsigned int i;
    unsigned int j;
    float d;
} SparseRow;

typedef struct node
{
    unsigned int n;   // Node label
    float value;       // Value of a general scalar quantity
    unsigned int childs_number;
    unsigned int * childs;
    float * distances;
    float * position; // Position in the embedding
} Node;

// Link structure is useless at the moment,
// may bind it to Linkin python later
typedef struct link
{
    Node * node1;
    Node * node2;
    float distance;
} Link;

typedef struct graph
{
    unsigned int N_nodes;
    unsigned long N_links;
    Node * nodes;
    unsigned int embedding_dimension;
} Graph;

// Global variables remain the same call after call
Graph G;
int RAND_INIT = 0;

// Link nodes in the Graph
void link_nodes(Node * node, unsigned int child_index, float distance){

    // Adds label of child to child array
    // printf("cnets - link_nodes - realloc childs (old = %p)...", node -> childs); fflush(stdout);
    node -> childs = (unsigned int *) realloc(node -> childs, sizeof(unsigned int)*( (node -> childs_number) + 1));
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
    node -> distances = (float *) realloc(node -> distances, sizeof(float)*((*node).childs_number + 1));
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

Graph to_Net(SparseRow * SM, float * values, unsigned int N_elements, unsigned long N_links){
    // printf("cnets - to_Net - ALLOC\n");
    Graph g;
    g.nodes = (Node *) malloc(sizeof(Node)*N_elements);
    if (g.nodes == NULL)
    {
        printf("!!! cannot allocate memory for %d nodes !!!\n", N_elements);
        exit(-1);
    }
    // printf("cnets - to_Net - ASSIGN\n");
    // Values assignment
    for (unsigned int k = 0; k < N_elements; k++)
    {
        g.nodes[k].n = k;
        g.nodes[k].value = values[k];
        g.nodes[k].childs_number = 0;
        g.nodes[k].childs = (unsigned int *) malloc(sizeof(unsigned int)); // Since each node has at least one child
        g.nodes[k].distances = (float *) malloc(sizeof(float));
    }

    // printf("cnets - to_Net - LINK\n");
    // Linking
    for (unsigned long k = 0; k < N_links; k++)
    {
        link_nodes(&(g.nodes[SM[k].i]), SM[k].j, SM[k].d);
        link_nodes(&(g.nodes[SM[k].j]), SM[k].i, SM[k].d);
    }
    // printf("cnets - to_Net - NEl&NLi\n");
    g.N_nodes = N_elements;
    g.N_links = N_links;
    return g;
}

float * PyList_to_double(PyObject * Pylist, unsigned int N_elements){

    float * dlist = (float *) malloc(sizeof(float)*N_elements);
    for (unsigned int i = 0; i < N_elements; i++)
    {
        dlist[i] = (float) PyFloat_AS_DOUBLE(PyList_GetItem(Pylist, i));
    }
    return dlist;
}

SparseRow * PyList_to_SM(PyObject * list, unsigned long N_links){

    SparseRow * SM = (SparseRow *) malloc(sizeof(SparseRow)*N_links);
    for (unsigned long k = 0; k < N_links;k++)
    {
        PyObject * row = PyList_GetItem(list,k);
        if (!PyList_Check(row))
        {
            printf("Invalid sparse list (row %li)\n", k);
            return NULL;
    }
    SM[k].i = (unsigned int) PyLong_AsLong(PyList_GetItem(row,0));
    SM[k].j = (unsigned int) PyLong_AsLong(PyList_GetItem(row,1));
    SM[k].d = (unsigned int) PyFloat_AsDouble(PyList_GetItem(row,2));
    }
    return SM;
}

unsigned int child_local_index_by_child_name(unsigned int node_number, unsigned int child_name)
{
    for (unsigned int child_local_index = 0; child_local_index < G.nodes[node_number].childs_number; child_local_index++)
        {
            if (G.nodes[node_number].childs[child_local_index] == child_name)
            {
                return child_local_index;
            }
        }
    return (unsigned int) -1;
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

    for (unsigned int n = 0; n < G.N_nodes; n++)
    {
        G.nodes[n].position = (float *) malloc(sizeof(float)*G.embedding_dimension);
        for (unsigned int d = 0; d < G.embedding_dimension; d++)
        {
            G.nodes[n].position[d] = ((float) rand())/((float) RAND_MAX);
        }
    }
    return;
}

// Python enters here first 90% of the time
PyObject * init_network(PyObject * self, PyObject * args){

    // The PyObjs for the two lists
    PyObject * Psparse = NULL;
    PyObject * Pvalues = NULL;
    unsigned int embedding_dim = 0;

    // The C sparse matrix and the values array
    SparseRow * SM = NULL;
    float * values = NULL;

    // Take the args and divide them in two pyobjects
    printf("cnets - Parsing...");fflush(stdout);
    if (!PyArg_ParseTuple(args,"OOi",&Psparse, &Pvalues, &embedding_dim)) Py_RETURN_NONE;
    printf("\tDone.\n");

    unsigned int N_elements = (unsigned int) PyList_Size(Pvalues);
    unsigned long N_links = (unsigned long) PyList_Size(Psparse);
    if (N_links < 1 ||  N_elements < 2)
    {
        printf("cnets - invalid network G = (%d, %ld)\n", N_elements, N_links);
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

float euclidean_distance(float * pos1, float * pos2, unsigned int dim){

    float dist = 0.;
    for (unsigned int i = 0; i < dim; i++)
    {
        dist += pow(pos1[i] - pos2[i], 2);
    }
    return sqrt(dist);
}

void move_away_from_random_not_child(unsigned int node, float eps){
    // Picks a guy at random until it is not a child
    unsigned int not_child;
    unsigned int draws = 0;
    do{
        not_child = (unsigned int)((G.N_nodes-1)*((float) rand()/RAND_MAX));
        draws++;
        if (draws > G.N_nodes){
            return;
        }
    }while(child_local_index_by_child_name(node, not_child) != (unsigned int)-1 || node == not_child);
    float dist = euclidean_distance(G.nodes[node].position, G.nodes[not_child].position, G.embedding_dimension);
    for (unsigned int d = 0; d < G.embedding_dimension; d++)
    {
        G.nodes[node].position[d] += eps/(dist*dist)*(G.nodes[node].position[d] - G.nodes[not_child].position[d]);
        if (G.nodes[node].position[d] != G.nodes[node].position[d]){
            printf("NAN detected\n");
            printf("%d- %d --> %lf\n", node, not_child, dist);
            exit(5);
        }
    }
    return;
}

PyObject * MDE(PyObject * self, PyObject * args){
    float eps = 0., neg_eps = 0.;
    unsigned int number_of_steps = 0;

    if (!PyArg_ParseTuple(args, "ffi", &eps, &neg_eps, &number_of_steps))
    {
        printf("cnets - ERROR parsing MDE args\n");
        Py_RETURN_NONE;
    }
    printf("cnets - starting MDE with eps = %lf, neg_eps = %lf, Nsteps = %d\n",eps, neg_eps, number_of_steps);

    float actual_distance = 0., factor;
    unsigned int child_index;
    for (unsigned int i = 0; i < number_of_steps; i++)
    {   
        progress_bar(((float)i)/( (float) number_of_steps) , 60);
        for (unsigned int current_node = 0; current_node < G.N_nodes; current_node++)
        {
            for (unsigned int current_child = 0; current_child < G.nodes[current_node].childs_number; current_child++ )
            {
                child_index = G.nodes[current_node].childs[current_child];
                actual_distance = euclidean_distance(G.nodes[current_node].position, G.nodes[child_index].position, G.embedding_dimension);                
                factor = eps*(1.- G.nodes[current_node].distances[current_child]/actual_distance)/G.nodes[current_node].childs_number;
                
                for (unsigned int d = 0; d < G.embedding_dimension; d++)
                {
                    G.nodes[current_node].position[d] += factor*(G.nodes[child_index].position[d] - G.nodes[current_node].position[d]) ;
                }
            }
            if (neg_eps != 0.){
                for (unsigned int mv_aw = 0; mv_aw < (unsigned int)(0.10*G.N_nodes); mv_aw++)
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
    for (unsigned int n = 0; n < G.N_nodes; n++)
    {
        PyObject * single = PyList_New(G.embedding_dimension);
        for (unsigned int d = 0; d < G.embedding_dimension; d++)
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

    PyObject * distanceSM = PyList_New(G.N_nodes*G.N_nodes); // Mmmh, not so clever! N**2 - > 9*N**2
    float d;
    long row_index = 0;
    Node node, another_node;

    for (unsigned int node_index = 0; node_index < G.N_nodes; node_index++)
    {   
        for (unsigned int another_node_index = 0; another_node_index < G.N_nodes; another_node_index++)
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

PyObject * matrix_to_list_of_list(float **mat, unsigned int N)
{
    /* Returns a matrix as list of lists (lol).
        Waiting to implement numpy arrays. */
    PyObject * lol = PyList_New(G.N_nodes);

    for (unsigned int i = 0; i < N; i++)
    {   
        PyObject * i_th_row = PyList_New(N);
        for (unsigned int j = 0; j < N; j++)
        {   
            PyList_SetItem(i_th_row, j , PyFloat_FromDouble((double) mat[i][j]));
        }
        PyList_SetItem(lol, i , i_th_row);  
    }
    return lol;
}

PyObject * get_distanceM(PyObject * self, PyObject * args)
{
    printf("cnets - getting distances...");fflush(stdout);
    /* Returns a matrix of distances as list of lists.
        Waiting to implement numpy arrays. */
    float ** distanceM = (float**) malloc(sizeof(float*)*G.N_nodes);
    for (unsigned int k=0; k< G.N_nodes; k ++)
    {
        distanceM[k] = (float*) malloc(sizeof(float)*G.N_nodes);
    }
    float d;
    Node node, another_node;

    for (unsigned int node_index = 0; node_index < G.N_nodes; node_index++)
    {   
        for (unsigned int another_node_index = node_index; another_node_index < G.N_nodes; another_node_index++)
        {   
            node = G.nodes[node_index];
            another_node = G.nodes[another_node_index];
            d = euclidean_distance(node.position, another_node.position, G.embedding_dimension);
            distanceM[node_index][another_node_index] = d;
            distanceM[another_node_index][node_index] = d;
        }        
    }
    printf("Done.\n");
    return matrix_to_list_of_list(distanceM, G.N_nodes);
}

PyObject * set_target(PyObject * self, PyObject * args)
{
    // printf("cnets - updating target distances\n");
    PyObject * PySM;
    unsigned int node1_number, node2_number;

    if(! PyArg_ParseTuple(args, "O", &PySM))
    {
        printf("cnets - set_target: paring failed\n");
    }
    SparseRow * SM = PyList_to_SM(PySM, G.N_links);
    for (unsigned long link = 0; link < G.N_links; link++)
    {
        node1_number = SM[link].i;
        node2_number = SM[link].j;
        // printf("cnet - set_target - (row %d of SM) --> (%d, %d) was (%lf), now is (%lf)\n", link, node1_number,node2_number, G.nodes[node1_number].distances[child_local_index_by_child_name(node1_number, node2_number)] , SM[link].d);
        
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
    {"get_distanceM", get_distanceM, METH_VARARGS,"Returns the distance matrix"},
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
