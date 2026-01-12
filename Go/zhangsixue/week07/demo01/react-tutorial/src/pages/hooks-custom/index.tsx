import React, { useState, useEffect } from "react";
import {
  Table,
  Form,
  Tag,
} from "antd";
import { ColumnsType } from "antd/es/table";
interface DataType {
  id: string;
  out_trade_no: string;
  order_status: string;
  is_send: boolean;
  count: number;
  ext2: string;
  price: number;
  from: string;
  c_openid: string;
  timeStamp: number;
  pay_level: string;
}

const useFetchData = (page: number, pageSize: number) => {
  const [data, setData] = useState<DataType[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [total, setTotal] = useState<number>(0);

  const fetchData = async () => {
    setLoading(true);
    try {
      const options = {
        pageIndex: page,
        pageSize: pageSize,
      };
      // Replace with your actual API endpoint
      const response = await fetch("/api/ten/getorderlist", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(options),
      });
      const result = await response.json();
      setData(result.data);
      setTotal(result.total);
    } catch (error) {
      console.error("Failed to fetch data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [page, pageSize]);

  return { data, loading, total, refreshData: fetchData };
};

function App() {
  const [form] = Form.useForm();

  const columns: ColumnsType<DataType> = [
    {
      title: "ID",
      dataIndex: "id",
      align: "center",
      key: "id",
      fixed: "left",
      render: (id, data) => {
        const today = new Date();
        const date = new Date(data.timeStamp * 1000);
        const isToday = date.toDateString() === today.toDateString();
        if (isToday) {
          return <span style={{ color: "#cf1322" }}>{id}</span>;
        }
        return id;
      },
    },
    {
      title: "类型",
      dataIndex: "test_type",
      align: "center",
      key: "test_type",
      fixed: "left",
    },
    {
      title: "订单号",
      dataIndex: "out_trade_no",
      align: "center",
      key: "out_trade_no",
    },
    {
      title: "订单状态",
      dataIndex: "order_status",
      align: "center",
      key: "order_status",
      render: (status) => {
        const statusMap: Record<string, any> = {
          "0": <Tag color="red">未支付</Tag>,
          "1": <Tag color="blue">已支付</Tag>,
        };
        return statusMap[status] || status;
      },
    },
    {
      title: "价格",
      dataIndex: "price",
      align: "center",
      key: "price",
      render: (price, data) => {
        if (data.ext2 == "wechat") {
          return price / 100 + "元";
        } else if (data.ext2 == "alipay") {
          return price + "元";
        } else {
          return price;
        }
      },
    },
    {
      title: "支付等级",
      dataIndex: "pay_level",
      align: "center",
      key: "pay_level",
    },
    {
      title: "支付方式",
      dataIndex: "ext2",
      align: "center",
      key: "ext2",
      render: (ext2) => {
        if (ext2 == "wechat") {
          return "微信";
        } else if (ext2 == "alipay") {
          return "支付宝";
        } else {
          return ext2;
        }
      },
    },
    {
      title: "openID",
      dataIndex: "c_openid",
      align: "center",
      key: "c_openid",
    },

    {
      title: "支付时间",
      align: "center",
      dataIndex: "timeStamp",
      key: "timeStamp",
      render: (text) => {
        const today = new Date();
        const date = new Date(text * 1000);
        const isToday = date.toDateString() === today.toDateString();
        if (isToday) {
          return (
            <span style={{ color: "#cf1322" }}>{date.toLocaleString()}</span>
          );
        }
        return new Date(text * 1000).toLocaleString();
      },
    },
    {
      title: "是否赠送",
      dataIndex: "is_send",
      align: "center",
      key: "is_send",
      render: (isSend) => (isSend ? "是" : "否"),
    },
  ];

  const [current, setCurrent] = useState(1);
  const [pageSize, setPageSize] = useState(15);
  const {
    data = [],
    loading,
    total,
    // refreshData,
  } = useFetchData(current, pageSize);
  const onChange = (page: number) => {
    setCurrent(page);
  };

  return (
    <div style={{ margin: "8px" }}>
      {loading ? (
        <div>loading...</div>
      ) : (
        data.length > 0 && (
          <>
            <div
              style={{ fontSize: "14px", textAlign: "right", margin: "10px" }}
            >
              总数据 : {total}条
            </div>
            <Table
              rowKey="out_trade_no"
              columns={columns}
              dataSource={data}
              size="small"
              scroll={{ x: "max-content" }}
              pagination={{
                current,
                total: total,
                pageSize,
                onChange,
                onShowSizeChange: (current, size) => {
                  setPageSize(size);
                  setCurrent(1);
                },
              }}
            />
          </>
        )
      )}
    </div>
  );
}

export default App;
