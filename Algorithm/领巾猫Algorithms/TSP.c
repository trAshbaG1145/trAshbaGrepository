#include <stdio.h>
#include <math.h>
#define MaxLen 1000
#define MaxVal 114514

typedef struct point
{
    double x;
    double y;
} point;
double dist(point p[], int a, int b)
{
    double dx = p[a].x - p[b].x;
    double dy = p[a].y - p[b].y;
    return sqrt(dx * dx + dy * dy);
}

int main()
{
    int num;
    scanf("%d", &num);
    if (num <= 1)
    {
        printf("0.00\n");
        return 0;
    }
    point p[num];
    for (int i = 0; i < num; i++)
    {
        scanf("%lf %lf", &p[i].x, &p[i].y);
    }
    double b[MaxLen][MaxLen];
    if (num >= 2)
    {
        b[0][1] = dist(p, 0, 1);
    }
    for (int j = 2; j < num; ++j)
    {
        for (int i = 0; i <= j - 2; ++i)
        {
            b[i][j] = b[i][j - 1] + dist(p, j - 1, j);
        }
        b[j - 1][j] = MaxVal;
        for (int k = 0; k <= j - 2; ++k)
        {
            double temp = b[k][j - 1] + dist(p, k, j);
            if (temp < b[j - 1][j])
            {
                b[j - 1][j] = temp;
            }
        }
    }
    double result;
    if (num == 2)
    {
        result = 2 * b[0][1];
    }
    else
    {
        result = b[num - 2][num - 1] + dist(p, num - 2, num - 1);
    }
    printf("%.2lf\n", result);
    return 0;
}