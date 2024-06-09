#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/* Read data from files */
void read_data(char *file_name, int n_feature, double **data, int *n_sample)
{
  FILE *fp = fopen(file_name, "r");
  int read_success = 1, max_sample = 100;
  double *sample = (double *)malloc((n_feature + 1) * sizeof(double));
  *n_sample = 0;
  *data = (double *)malloc(max_sample * (n_feature + 1) * sizeof(double));

  while (read_success) {
    int feature_idx = 0;
    while (read_success && feature_idx <= n_feature) {
      if (fscanf(fp, "%lf", sample + feature_idx) == EOF) {
        read_success = 0;
        fclose(fp);
      }
      feature_idx++;
    }

    if (read_success) {
      if (*n_sample >= max_sample) {
        max_sample = max_sample + 100;
        *data = (double *)realloc(*data, max_sample * (n_feature + 1) * sizeof(double));
      }
      memcpy((*data) + (*n_sample) * (n_feature + 1), sample, (n_feature + 1) * sizeof(double));
      (*n_sample)++;
    }
  }

  free(sample);
}

/* Local summation */
void summation(double *data, int n_ele, int n_sample, double *sum)
{
  int i, n;
  for (i = 0; i < n_ele; i++) sum[i] = 0;
  for (n = 0; n < n_sample; n++)
    for (i = 0; i < n_ele; i++, data++)
      sum[i] = sum[i] + *data;
}

/* Global averaging */
void average(double *sum, int n_ele, int n_sample, int rank, double *avg)
{
  int i, n;
  MPI_Allreduce(sum, avg, n_ele, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&n_sample, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  for (i = 0; i < n_ele; i++) avg[i] /= n;
}

/* Compute the local part of the coefficient matrix and the right-hand side */
void weight_coef(double *data, int n_feature, int n_sample, double *avg, double *sum)
{
  int i, j, n, len = n_feature * (n_feature + 3) / 2;
  for (i = 0; i < len; i++) sum[i] = 0;

  double *p_data = data;
  for (n = 0; n < n_sample; n++, p_data += n_feature + 1) {
    for (i = 0; i < n_feature; i++)
      sum[i] += p_data[n_feature] * (p_data[i] - avg[i]);

    for (i = 0; i < n_feature; i++) {
      len = n_feature + i*(i+1)/2;
      for (j = 0; j <= i; j++)
        sum[len + j] += (p_data[i] - avg[i]) * (p_data[j] - avg[j]);
    }
  }
}

/* Use LDL^T decomposition to solve the linear system */
void solve_spd_system(int size, double *matrix, double *rhs, double *solution)
{
  int i, j, k;

  /* LDL^T decomposition */
  for (i = 0; i < size; i++) {
    int len_i = i*(i+1)/2;
    for (j = 0; j <= i; j++) {
      int len_j = j*(j+1)/2;
      double sum = 0;
      for (k = 0; k < j; k++)
        sum += matrix[len_i + k] * matrix[len_j + k] * matrix[k*(k+3)/2];

      if (i == j)
        matrix[len_i + j] = matrix[len_i + i] - sum;
      else
        matrix[len_i + j] = (1.0 / matrix[len_j + j] * (matrix[len_i + j] - sum));
    }
  }

  /* Forward substitution */
  for (i = 1; i < size; i++) {
    int len_i = i*(i+1)/2;
    for (j = 0; j < i; j++)
      rhs[i] -= matrix[len_i + j] * rhs[j];
  }

  /* Solve the diagonal system */
  for (i = 0; i < size; i++) solution[i] = rhs[i] / matrix[i * (i+3)/2];

  /* Backward substitution */
  for (i = size-1; i > 0; i--) {
    int len_i = i*(i+1)/2;
    for (j = 0; j < i; j++)
      solution[j] -= matrix[len_i + j] * solution[i];
  }
}

/* Compute weights of the linear system */
void compute_weight(double *sum, double* avg, int n_feature, int n_sample, int nprocs, int rank, double *weight)
{
  int len = n_feature * (n_feature + 3) / 2;
  if (rank == 0) {
    int i;
    MPI_Reduce(MPI_IN_PLACE, sum, len, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    solve_spd_system(n_feature, sum + n_feature, sum, weight);

    weight[n_feature] = avg[n_feature];
    for (i = 0; i < n_feature; i++)
      weight[n_feature] -= weight[i] * avg[i];

    MPI_Bcast(weight, n_feature + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(sum, NULL, len, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(weight, n_feature + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

/* Compute the RMSE */
double test(double *data, int n_feature, int n_sample, int rank, double *weight) {
  int i, j;
  double squared_err = 0, avg;
  for (i = 0; i < n_sample; i++) {
    double diff = 0;
    for (j = 0; j < n_feature; j++, data++)
      diff += *data * weight[j];
    diff += weight[n_feature] - *(data++);
    squared_err += diff * diff;
  }
  average(&squared_err, 1, n_sample, rank, &avg);
  return sqrt(avg);
}

int main(int argc, char *argv[])
{
  int nprocs, rank, n_sample, n_train, n_test, n_feature, i;
  char file_name[100];
  double *data, *sum, *avg, *weight, rmse;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (argc < 3) {
    if (rank == 0)
      fprintf(stderr, "Usage: %s <Number of Features> <File Name Prefix>\n", argv[0]);
    MPI_Finalize();
    return 1;
  }
  n_feature = atoi(argv[1]);

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  sprintf(file_name, "%s_%d.dat", argv[2], rank);
  read_data(file_name, n_feature, &data, &n_sample);

  /* Split the data into the training set (70%) and the test set (30%) */
  n_train = (int)ceil(n_sample * 0.7); n_test = n_sample - n_train;

  sum = (double *)malloc(sizeof(double) * (n_feature + 3) * n_feature / 2);
  summation(data, n_feature + 1, n_train, sum);

  avg = (double *)malloc(sizeof(double) * (n_feature + 1));
  average(sum, n_feature + 1, n_train, rank, avg);

  weight = (double *)malloc(sizeof(double) * (n_feature + 1));
  weight_coef(data, n_feature, n_train, avg, sum);
  compute_weight(sum, avg, n_feature, n_train, nprocs, rank, weight);
  if (rank == 0) {
    printf("Weights: [%lf", weight[0]);
    for (i = 1; i <= n_feature; i++) printf(", %lf", weight[i]);
    printf("]\n");
  }

  rmse = test(data, n_feature, n_train, rank, weight);
  if (rank == 0)
    printf("Root mean squared error for training data: %lf\n", rmse);

  rmse = test(data + (n_feature+1) * n_train, n_feature, n_test, rank, weight);
  if (rank == 0)
    printf("Root mean squared error for test data: %lf\n", rmse);

  free(data);
  free(sum);
  free(avg);
  free(weight);

  MPI_Finalize();
}
