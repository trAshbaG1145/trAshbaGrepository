#include <stdio.h>
int pow_10(int n) 
{
    int result = 1;
    for (int i = 0; i < n; i++) 
    {
        result *= 10;
    }
    return result;
}
int find_max(int array[], int size) 
{
    int max = array[0];
    for (int i = 1; i < size; i++) 
    {
        if (array[i] > max) {
            max = array[i];
        }
    }
    return max;
}
int get_digits(int num) 
{
    if (num == 0) return 1;
    int count = 0;
    while (num != 0) 
    {
        num /= 10;
        count++;
    }
    return count;
}
int get_digit(int num, int d) 
{
    return (num / pow_10(d)) % 10;
}
void radix_sort(int array[], int size) 
{
    if (size <= 1) return;
    int max = find_max(array, size);
    int max_digits = get_digits(max);
    int buckets[10][size];
    int bucket_counts[10] = {0};
    for (int d = 0; d < max_digits; d++) 
    {
        for (int i = 0; i < 10; i++) 
        {
            bucket_counts[i] = 0;
        }
        for (int i = 0; i < size; i++) 
        {
            int digit = get_digit(array[i], d);
            buckets[digit][bucket_counts[digit]++] = array[i];
        }
        int index = 0;
        for (int i = 0; i < 10; i++) 
        {
            for (int j = 0; j < bucket_counts[i]; j++) 
            {
                array[index++] = buckets[i][j];
            }
        }
    }
}
void print_array(int array[], int size) 
{
    for (int i = 0; i < size; i++) 
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}
int main() 
{
    int n;
    scanf("%d", &n);
    int array[n];
    for (int i = 0; i < n; i++) 
    {
        scanf("%d", &array[i]);
    }
    radix_sort(array, n);
    print_array(array, n);
    return 0;
}