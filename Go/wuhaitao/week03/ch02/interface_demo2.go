package main

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"time"
)

// HTTPClient 定义了HTTP客户端的接口
type HTTPClient interface {
	Get(url string) ([]byte, error)
	Post(url string, contentType string, body []byte) ([]byte, error)
}

// 实际HTTP客户端实现
type RealHTTPClient struct {
	client *http.Client
}

// NewRealHTTPClient 创建一个真实的HTTP客户端
func NewRealHTTPClient(timeout time.Duration) *RealHTTPClient {
	return &RealHTTPClient{
		client: &http.Client{
			Timeout: timeout,
		},
	}
}

// Get 实现接口的Get方法
func (c *RealHTTPClient) Get(url string) ([]byte, error) {
	resp, err := c.client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return io.ReadAll(resp.Body)
}

// Post 实现接口的Post方法
func (c *RealHTTPClient) Post(url string, contentType string, body []byte) ([]byte, error) {
	resp, err := c.client.Post(url, contentType, bytes.NewBuffer(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return io.ReadAll(resp.Body)
}

// 模拟HTTP客户端实现，用于测试
type MockHTTPClient struct {
	GetResponse  []byte
	PostResponse []byte
	GetError     error
	PostError    error
}

// Get 模拟Get请求
func (m *MockHTTPClient) Get(url string) ([]byte, error) {
	return m.GetResponse, m.GetError
}

// Post 模拟Post请求
func (m *MockHTTPClient) Post(url string, contentType string, body []byte) ([]byte, error) {
	return m.PostResponse, m.PostError
}

// 使用HTTP客户端的服务
type APIService struct {
	client HTTPClient
}

// NewAPIService 创建API服务
func NewAPIService(client HTTPClient) *APIService {
	return &APIService{
		client: client,
	}
}

// FetchData 从API获取数据
func (s *APIService) FetchData(url string) (string, error) {
	data, err := s.client.Get(url)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// SendData 发送数据到API
func (s *APIService) SendData(url string, data string) (string, error) {
	resp, err := s.client.Post(url, "application/json", []byte(data))
	if err != nil {
		return "", err
	}
	return string(resp), nil
}

func main() {
	// 使用真实HTTP客户端
	realClient := NewRealHTTPClient(10 * time.Second)
	fmt.Println("创建了真实HTTP客户端，超时设置为:", realClient.client.Timeout)

	// 注释掉以下代码，因为需要网络连接，仅作为示例
	// realService := NewAPIService(realClient)
	// result, err := realService.FetchData("https://example.com")

	// 使用模拟客户端进行测试
	mockClient := &MockHTTPClient{
		GetResponse:  []byte(`{"status": "success", "data": "示例数据"}`),
		PostResponse: []byte(`{"status": "success", "message": "数据已接收"}`),
		GetError:     nil,
		PostError:    nil,
	}
	mockService := NewAPIService(mockClient)

	// 使用模拟客户端测试
	result, err := mockService.FetchData("https://example.com")
	if err != nil {
		fmt.Printf("获取数据错误: %v\n", err)
		return
	}

	fmt.Printf("获取的数据: %s\n", result)

	// 示例：发送数据
	sendResult, err := mockService.SendData(
		"https://example.com/api",
		`{"name": "测试", "value": 123}`,
	)
	if err != nil {
		fmt.Printf("发送数据错误: %v\n", err)
		return
	}

	fmt.Printf("发送数据响应: %s\n", sendResult)
}
