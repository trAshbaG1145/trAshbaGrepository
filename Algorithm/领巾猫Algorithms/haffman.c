#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

typedef struct node
{
    int value;
    struct node *Rchild;
    struct node *Lchild;
} node;
void findTwoMin(node *nodes[], int n, int *min1, int *min2)
{
    int min_val1 = INT_MAX, min_val2 = INT_MAX;
    *min1 = *min2 = -1;

    for (int i = 0; i < n; i++)
    {
        if (nodes[i] != NULL && nodes[i]->value < min_val1)
        {
            min_val2 = min_val1;
            *min2 = *min1;
            min_val1 = nodes[i]->value;
            *min1 = i;
        }
        else if (nodes[i] != NULL && nodes[i]->value < min_val2)
        {
            min_val2 = nodes[i]->value;
            *min2 = i;
        }
    }
}
node *Create(int w[], int n)
{
    if (n <= 0)
        return NULL;
    node **nodes = (node **)malloc(n * sizeof(node *));
    for (int i = 0; i < n; i++)
    {
        nodes[i] = (node *)malloc(sizeof(node));
        nodes[i]->value = w[i];
        nodes[i]->Lchild = NULL;
        nodes[i]->Rchild = NULL;
    }
    int remaining = n;
    while (remaining > 1)
    {
        int min1, min2;
        findTwoMin(nodes, n, &min1, &min2);
        node *newNode = (node *)malloc(sizeof(node));
        newNode->value = nodes[min1]->value + nodes[min2]->value;
        newNode->Lchild = nodes[min1];
        newNode->Rchild = nodes[min2];
        nodes[min1] = newNode;
        nodes[min2] = NULL;
        remaining--;
    }
    node *root = NULL;
    for (int i = 0; i < n; i++)
    {
        if (nodes[i] != NULL)
        {
            root = nodes[i];
            break;
        }
    }

    free(nodes);
    return root;
}
int calculateWPL(node *root, int depth)
{
    if (root == NULL)
        return 0;
    if (root->Lchild == NULL && root->Rchild == NULL)
        return root->value * depth;
    return calculateWPL(root->Lchild, depth + 1) + calculateWPL(root->Rchild, depth + 1);
}
int main()
{
    int n;
    scanf("%d", &n);
    int value[n];
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &value[i]);
    }
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (value[j] > value[j + 1])
            {
                int temp = value[j];
                value[j] = value[j + 1];
                value[j + 1] = temp;
            }
        }
    }
    node *root = Create(value, n);
    printf("%d\n", calculateWPL(root, 0));
    return 0;
}