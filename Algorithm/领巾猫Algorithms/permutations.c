#include <stdio.h>
void print_array(int array[], int size) 
{
    for (int i = 0; i < size; i++) 
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}
void swap(int *x, int *y)
{
    int temp = *x;
    *x = *y;
    *y = temp;
}
void generate_permutations(int start, int end, int array[])
{
    if (start == end)
    {
        print_array(array, end + 1);
        return;
    }
    for (int i = start; i <= end; i++)
    {
        swap(&array[start], &array[i]);
        generate_permutations(start + 1, end, array);
        swap(&array[start], &array[i]);
    }
}

int main()
{
    int n;
    scanf("%d", &n);
    int array[n];
    for (int i = 0; i < n; i++)
    {
        array[i] = i + 1;
    }
    generate_permutations(0, n - 1, array);
    return 0;
}
