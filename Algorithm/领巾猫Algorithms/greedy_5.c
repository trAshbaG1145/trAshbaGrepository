#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>

int comp(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

int main()
{
    int n, L;
    scanf("%d %d", &n, &L);
    int program[n];
    int current = 0;
    int num = 0;
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &program[i]);
    }
    int k = sizeof(program) / sizeof(program[0]);
    qsort(program, k, sizeof(program[0]), comp);
    for (int i = 0; i < n; i++)
    {
        if (current + program[i] > L)
        {
            break;
        }
        else
        {
            current += program[i];
            num++;
        }
    }
    printf("%d", num);
}