#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define MAX_VERTICES 100
#define INF INT_MAX

void floyd(int n, int graph[MAX_VERTICES][MAX_VERTICES], int dist[MAX_VERTICES][MAX_VERTICES])
{
    int i, j, k;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            dist[i][j] = graph[i][j];
        }
    }
    for (k = 0; k < n; k++)
    {
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                if (dist[i][k] != INF && dist[k][j] != INF)
                {
                    if (dist[i][j] > dist[i][k] + dist[k][j])
                    {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }
    }
}

int main()
{
    int n, m;
    int graph[MAX_VERTICES][MAX_VERTICES];
    int dist[MAX_VERTICES][MAX_VERTICES];
    int vertices[MAX_VERTICES];
    scanf("%d %d", &n, &m);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                graph[i][j] = 0;
            }
            else
            {
                graph[i][j] = INF;
            }
        }
    }
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &vertices[i]);
    }
    for (int i = 0; i < m; i++)
    {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        graph[u - 1][v - 1] = w;
    }
    floyd(n, graph, dist);
    for (int i = 1; i < n; i++)
    {
        if (dist[0][i] == INF)
        {
            printf("INF\n");
        }
        else
        {
            printf("%d\n", dist[0][i]);
        }
    }
    return 0;
}