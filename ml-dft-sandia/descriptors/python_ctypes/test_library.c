#include <stdio.h>
#include <stdlib.h>
#include "test_library.h"

void hello_world() {
  printf(" Hello, world!\n");
}

double** create_matrix(int n1, int n2) {
  double **array;
  double *values;

  values = (double*) malloc(n1*n2*sizeof(double));
  array = (double**) malloc(n1*sizeof(double*));
  
  int n = 0;
  for (int i = 0; i < n1; i++) {
    array[i] = &values[n];
    n += n2;
  }

  double fac = 1.0/(n1*(n2+1));
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      array[i][j] = fac*(i+1)*(j+1+1);
    }
  }

  return array;
}

void destroy_matrix(double** array) {
  free(array[0]);
  free(array);
}