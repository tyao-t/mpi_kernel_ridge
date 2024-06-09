#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define N_FEATURE 1

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

void summation(double *data, int n_feature, int n_sample, double *sum)
{
  int i, n;
  for (i = 0; i <= n_feature; i++)
    sum[i] = 0;

  for (n = 0; n < n_sample; n++)
    for (i = 0; i <= n_feature; i++, data++) {
      sum[i] = sum[i] + *data;
    }
}

void average(double *sum, int n_feature, int n_sample, int nprocs, int rank, double *avg)
{
  int i, n;
  MPI_Allreduce(sum, avg, n_feature + 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&n_sample, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  for (i = 0; i <= n_feature; i++) avg[i] /= n;
}

void weight_coef(double *data, int n_feature, int n_sample, double *avg, double *sum)
{
  int i, n;
  for (i = 0; i <= n_feature; i++)
    sum[i] = 0;

  double *p_data = data;
  for (n = 0; n < n_sample; n++, p_data += n_feature + 1) {
    for (i = 0; i < n_feature; i++) {
      sum[i] += p_data[n_feature] * (p_data[i] - avg[i]);
      sum[n_feature] += (p_data[i] - avg[i]) * (p_data[i] - avg[i]);
    }
  }
}

void compute_weight(double *sum, double* avg, int n_feature, int n_sample, int nprocs, int rank, double *weight)
{
  if (rank == 0) {
    int i;
    MPI_Reduce(sum, weight, n_feature + 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double b = avg[n_feature];
    for (i = 0; i < n_feature; i++) {
      weight[i] /= weight[n_feature];
      b -= weight[i] * avg[i];
    }
    weight[n_feature] = b;

    MPI_Bcast(weight, n_feature + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(sum, weight, n_feature + 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(weight, n_feature + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

double test(double *data, int n_feature, int n_sample, int nprocs, int rank, double *weight) {
  int i, j;
  double squared_err = 0, avg;
  for (i = 0; i < n_sample; i++) {
    double diff = 0;
    for (j = 0; j < n_feature; j++, data++)
      diff += *data * weight[j];
    diff += weight[n_feature] - *(data++);
    squared_err += diff * diff;
  }
  average(&squared_err, 0, n_sample, nprocs, rank, &avg);
  return sqrt(avg);
}

int main(int argc, char *argv[])
{
  int nprocs, rank, n_sample, i;
  char file_name[100];
  double *data, *sum, *avg, weight[N_FEATURE+1], rmse;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  sprintf(file_name, "data_%d.dat", rank);
  read_data(file_name, N_FEATURE, &data, &n_sample);

  sum = (double *)malloc(sizeof(double) * (N_FEATURE + 1));
  summation(data, N_FEATURE, n_sample, sum);

  avg = (double *)malloc(sizeof(double) * (N_FEATURE + 1));
  average(sum, N_FEATURE, n_sample, nprocs, rank, avg);

  weight_coef(data, N_FEATURE, n_sample, avg, sum);
  compute_weight(sum, avg, N_FEATURE, n_sample, nprocs, rank, weight);
  if (rank == 0) {
    printf("Weights: [%lf", weight[0]);
    for (i = 1; i <= N_FEATURE; i++) printf(", %lf", weight[i]);
    printf("]\n");
  }

  rmse = test(data, N_FEATURE, n_sample, nprocs, rank, weight);
  if (rank == 0)
    printf("Root mean squared error: %lf\n", rmse);

  free(data);
  free(sum);
  free(avg);

  MPI_Finalize();
}
