#include <stdio.h>
#include <stdlib.h>

int dx[4] = {-1, 0, 1, 0};
int dy[4] = {0, 1, 0, -1};

typedef struct Node
{
    int x, y;
    int steps;
    struct Node *parent;
} Node;

int main()
{
    int n, m;
    scanf("%d %d", &n, &m);

    int posx, posy;
    scanf("%d %d", &posx, &posy);

    int finalx, finaly;
    scanf("%d %d", &finalx, &finaly);

    int matrix[n][m];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            scanf("%d", &matrix[i][j]);
        }
    }

    int visited[n][m];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            visited[i][j] = 0;
        }
    }
    Node *queue = (Node *)malloc((n * m) * sizeof(Node));
    int front = 0, rear = 0;
    queue[rear].x = posx - 1;
    queue[rear].y = posy - 1;
    queue[rear].steps = 1;
    queue[rear].parent = NULL;
    rear++;
    visited[posx - 1][posy - 1] = 1;

    Node *finalNode = NULL;

    while (front < rear)
    {
        Node current = queue[front];
        front++;
        if (current.x == finalx - 1 && current.y == finaly - 1)
        {
            finalNode = &queue[front - 1];
            break;
        }

        for (int i = 0; i < 4; i++)
        {
            int nx = current.x + dx[i];
            int ny = current.y + dy[i];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m &&
                matrix[nx][ny] == 0 && !visited[nx][ny])
            {
                visited[nx][ny] = 1;
                queue[rear].x = nx;
                queue[rear].y = ny;
                queue[rear].steps = current.steps + 1;
                queue[rear].parent = &queue[front - 1];
                rear++;
            }
        }
    }
    if (finalNode != NULL)
    {
        printf("%d\n", finalNode->steps);
    }
    else
    {
        printf("0\n");
    }
    free(queue);
    return 0;
}