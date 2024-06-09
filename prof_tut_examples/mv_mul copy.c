#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int rank; // Rank of the current process
  int n; // Size of the vector
  int* n_cpnt; // Number of components in process
  int* loc; // Location of the first element in each process
} size_info_t;

void create_size_info(int n, size_info_t *size)
{
  int i, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &size->rank);

  size->n_cpnt = (int *)malloc(sizeof(int) * nprocs);
  size->loc = (int *)malloc(sizeof(int) * nprocs);
  MPI_Allgather(&n, 1, MPI_INT, size->n_cpnt, 1, MPI_INT, MPI_COMM_WORLD);

  size->loc[0] = 0; size->n = size->n_cpnt[0];
  for (i = 1; i < nprocs; i++) {
    size->loc[i] = size->loc[i-1] + size->n_cpnt[i-1];
    size->n += size->n_cpnt[i];
  }
}

void destroy_size_info(size_info_t *size)
{
  free(size->n_cpnt);
  free(size->loc);
}

void mv_mul(const double *matrix, int n_row,
    const double *vector, const size_info_t *size, double *product)
{
  int i, j;
  double *whole_vector = (double *)malloc(size->n * sizeof(double));
  MPI_Allgatherv(vector, size->n_cpnt[size->rank], MPI_DOUBLE,
      whole_vector, size->n_cpnt, size->loc, MPI_DOUBLE, MPI_COMM_WORLD);
  for (i = 0; i < n_row; i++) {
    product[i] = 0;
    for (j = 0; j < size->n; j++)
      product[i] += matrix[i*size->n + j] * whole_vector[j];
  }
  free(whole_vector);
}

void print_result(FILE *fp, const double *matrix, int n_row,
    const double *vector, const size_info_t *size, const double *product)
{
  int i, j;
  fprintf(fp, "Matrix:\n");
  for (i = 0; i < n_row; i++) {
    fprintf(fp, "\t[");
    for (j = 0; j < size->n - 1; j++, matrix++)
      fprintf(fp, "%lf\t", *matrix);
    fprintf(fp, "%lf]\n", *(matrix++));
  }

  fprintf(fp, "\nVector:\n\t[");
  for (i = 0; i < size->n_cpnt[size->rank] - 1; i++, vector++)
    fprintf(fp, "%lf\n\t ", *vector);
  fprintf(fp, "%lf]\n", *(vector++));

  fprintf(fp, "\nProduct:\n\t[");
  for (i = 0; i < n_row - 1; i++, product++)
    fprintf(fp, "%lf\n\t ", *product);
  fprintf(fp, "%lf]\n", *(product++));
}

int main(int argc, char *argv[])
{
  int i, rank, n_row, n;
  size_info_t size;
  double *matrix, *vector, *product;
  char file_name[100];
  FILE *fp;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  srand(time(NULL) + rank);

  /* Number of rows of the matrix in the current process. */
  n_row = rand() % 10 + 1;

  /* Number of components of the matrix in the current process. */
  n = rand() % 10 + 1;

  /* Gather the size information from all processes. */
  create_size_info(n, &size);

  /* Generate a random matrix. */
  matrix = (double *)malloc(n_row * size.n * sizeof(double));
  for (i = 0; i < n_row * size.n; i++)
    matrix[i] = 1. * rand() / RAND_MAX;

  /* Generate a random vector. */
  vector = (double *)malloc(n * sizeof(double));
  for (i = 0; i < n; i++)
    vector[i] = 1. * rand() / RAND_MAX;

  /* Perform matrix-vector multiplication. */
  product = (double *)malloc(n_row * sizeof(double));
  mv_mul(matrix, n_row, vector, &size, product);

  /* Print result. */
  sprintf(file_name, "mv_mul_output_%d.dat", rank);
  fp = fopen(file_name, "w");
  print_result(fp, matrix, n_row, vector, &size, product);
  fclose(fp);

  /* Clean up */
  free(matrix);
  free(vector);
  free(product);
  destroy_size_info(&size);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
