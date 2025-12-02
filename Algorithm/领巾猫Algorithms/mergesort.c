#include <stdio.h>
#include <stdlib.h>
void Merge(int A[], int low, int mid, int high)
{
    int n1 = mid - low + 1;
    int n2 = high - mid;
    int L[n1];
    int R[n2];
    for (int i = 0; i < n1; i++)
        L[i] = A[low + i];
    for (int j = 0; j < n2; j++)
        R[j] = A[mid + 1 + j];
    int i = 0, j = 0, k = low;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            A[k++] = L[i++];
        }
        else
        {
            A[k++] = R[j++];
        }
    }
    while (i < n1)
        A[k++] = L[i++];
    while (j < n2)
        A[k++] = R[j++];
}
void mergesort(int A[], int low, int high)
{
    if (low < high)
    {
        int mid = (low + high) / 2;
        mergesort(A, low, mid);
        mergesort(A, mid + 1, high);
        Merge(A, low, mid, high);
    }
}
int main()
{
    int num;
    scanf("%d", &num);
    int A[num];
    for (int i = 0; i < num; i++)
    {
        scanf("%d", &A[i]);
    }
    mergesort(A, 0, num - 1);
    for (int i = 0; i < num; i++)
    {
        printf("%d ", A[i]);
    }
    return 0;
}