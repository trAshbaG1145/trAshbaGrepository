typedef struct node
{
    int value;                // 当前总价值
    int weight;               // 当前总重量
    bool check;               // 是否超重（超重=true，停止建树）
    struct node *in_left;     // 左子树：放入当前物品
    struct node *notin_right; // 右子树：不放入当前物品
} node;