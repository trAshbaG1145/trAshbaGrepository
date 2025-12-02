#include <stdio.h>
int select(int num[], int len, int k)
{
    if (len < 10)
    {
        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < len - i - 1; j++)
            {
                if (num[j] > num[j + 1])
                {
                    int temp = num[j];
                    num[j] = num[j + 1];
                    num[j + 1] = temp;
                }
            }
        }
        if (k < 0 || k >= len)
            return -1;
        return num[k];
    }
    int q = (len + 4) / 5;
    int M[q];
    for (int i = 0; i < q; i++)
    {
        int group[5], cnt = 0;
        for (int j = 0; j < 5 && i * 5 + j < len; j++)
        {
            group[cnt++] = num[i * 5 + j];
        }
        for (int m = 0; m < cnt; m++)
        {
            for (int n = 0; n < cnt - m - 1; n++)
            {
                if (group[n] > group[n + 1])
                {
                    int temp = group[n];
                    group[n] = group[n + 1];
                    group[n + 1] = temp;
                }
            }
        }
        M[i] = group[cnt / 2];
    }
    int mm = select(M, q, q / 2);
    int A1[len], A3[len];
    int x = 0, z = 0, y = 0;
    for (int i = 0; i < len; i++)
    {
        if (num[i] < mm)
        {
            A1[x++] = num[i];
        }
        else if (num[i] > mm)
        {
            A3[z++] = num[i];
        }
        else
        {
            y++;
        }
    }
    if (k < x)
    {
        return select(A1, x, k);
    }
    else if (k < x + y)
    {
        return mm;
    }
    else
    {
        return select(A3, z, k - x - y);
    }
}

int main()
{
    int n;
    scanf("%d", &n);
    int num[n];
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &num[i]);
    }
    int k;
    scanf("%d", &k);
    int res = select(num, n, k - 1);
    printf("%d\n", res);
    return 0;
}