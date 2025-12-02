#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define max(a, b) ((a) > (b) ? (a) : (b))
int n;
int *t1, *t2;
int best_sum;
int *best_order;
void buildScheduleTree(int used, int *order, int k, int f1, int f2, int sum_f2)
{
    if (k == n)
    {
        if (sum_f2 < best_sum)
        {
            best_sum = sum_f2;
            for (int i = 0; i < n; i++)
            {
                best_order[i] = order[i];
            }
        }
        return;
    }
    if (sum_f2 >= best_sum)
    {
        return;
    }
    for (int i = 0; i < n; i++)
    {
        if (!(used & (1 << i)))
        {
            int new_f1 = f1 + t1[i];
            int new_f2 = max(new_f1, f2) + t2[i];
            int new_sum = sum_f2 + new_f2;
            int new_order[n];
            for (int j = 0; j < k; j++)
            {
                new_order[j] = order[j];
            }
            new_order[k] = i;
            buildScheduleTree(used | (1 << i), new_order, k + 1, new_f1, new_f2, new_sum);
        }
    }
}

int main()
{
    scanf("%d", &n);
    t1 = (int *)malloc(n * sizeof(int));
    t2 = (int *)malloc(n * sizeof(int));
    best_order = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &t1[i]);
    }
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &t2[i]);
    }
    best_sum = INT_MAX;
    int init_order[0];
    buildScheduleTree(0, init_order, 0, 0, 0, 0);
    for (int i = 0; i < n; i++)
    {
        printf("%d", best_order[i] + 1);
        if (i != n - 1)
            printf(" ");
    }
    printf("\n");
    printf("%d\n", best_sum);
    return 0;
}