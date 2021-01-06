#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//

double abs_func(double value)
{
  if (value>0)
  {
    return value;
  }
  else
  {
    return -value;
  }
}

void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  double sum = 0;

  #pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob; //初始PageRank
    if (outgoing_size(g, i)==0)
    {
      sum += solution[i];
    }
    
  }

  double* score_new = new double [numNodes]; //新的PageRank


  bool converged = 0;
  while (!converged)
  {
    #pragma omp parallel for
    for (int i = 0; i < numNodes; i++)
    {
        
        score_new[i] = 0;
        const Vertex* first_vertex = incoming_begin(g, i); //第一個連進的vertex
        const Vertex* last_vertex = incoming_end(g, i); //最後一個連進的vertex

        for (const Vertex *current_vertex = first_vertex; current_vertex != last_vertex; current_vertex++)
        {
          score_new[i] += solution[*current_vertex]/(int)outgoing_size(g, *current_vertex); //先算出每個連進vertex的PageRank加總除以數量
        }
        
        score_new[i] = (score_new[i]*damping) + (1.0-damping)/numNodes; //再計算damping
        score_new[i] += damping*sum / numNodes;
    }

    sum = 0;
    double global_diff = 0; 

    #pragma omp parallel for reduction(+ : global_diff), reduction(+ : sum)
    for (int i = 0; i < numNodes; i++)
    {
      global_diff += abs_func(score_new[i]-solution[i]); //新舊PageRank相差的絕對值
      solution[i] = score_new[i];
      if (outgoing_size(g,i)==0)
      {
        sum += solution[i];
      }
      
    }

    if (global_diff<convergence) //若相差值小於收斂常數，就等於收斂
    {
      converged = 1;
    }
    
  }
  delete [] score_new;
  
  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
