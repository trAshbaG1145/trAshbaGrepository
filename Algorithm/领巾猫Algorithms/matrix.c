#include <stdio.h>
int main()
{
    int m, n, k;
    scanf("%d %d %d", &n, &m, &k);
    int A1[n][m];
    int A2[m][k];
    for (int i = 0; i < n; i++)
    {
        for (int t = 0; t < m; t++)
        {
            scanf("%d", &A1[i][t]);
        }
    }
    for (int i = 0; i < m; i++)
    {
        for (int t = 0; t < k; t++)
        {
            scanf("%d", &A2[i][t]);
        }
    }
    int C[n][k];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            C[i][j] = 0;
            for (int t = 0; t < m; t++)
            {
                C[i][j] += A1[i][t] * A2[t][j];
            }
        }
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }
}