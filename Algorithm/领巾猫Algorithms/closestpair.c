#include <stdio.h>
int main()
{
    int num;
    scanf("%d", &num);
    int loc[num][2];
    for (int i = 0; i < num; i++)
    {
        scanf("%d %d", &loc[i][0], &loc[i][1]);
    }
    for (int i = 0; i < num; i++)
    {
        for (int k = 0; k < num - i - 1; k++)
        {
            if (loc[k][0] > loc[k + 1][0])
            {
                int temp;
                temp = loc[k][0];
                loc[k][0] = loc[k + 1][0];
                loc[k + 1][0] = temp;
            }
        }
    }
    int pos = loc[num / 2][0];
    int xtotal = 0;
    for (int i = 0; i < num; i++)
    {
        int diff = pos - loc[i][0];
        if (diff < 0)
        {
            xtotal += -diff;
        }
        else
        {
            xtotal += diff;
        }
    }
    for (int i = 0; i < num; i++)
    {
        for (int k = 0; k < num - i - 1; k++)
        {
            if (loc[k][1] > loc[k + 1][1])
            {
                int temp;
                temp = loc[k][1];
                loc[k][1] = loc[k + 1][1];
                loc[k + 1][1] = temp;
            }
        }
    }
    pos = loc[num / 2][1];
    int ytotal = 0;
    for (int i = 0; i < num; i++)
    {
        int diff = pos - loc[i][1];
        if (diff < 0)
        {
            ytotal += -diff;
        }
        else
        {
            ytotal += diff;
        }
    }
    printf("%d\n", xtotal + ytotal);
}