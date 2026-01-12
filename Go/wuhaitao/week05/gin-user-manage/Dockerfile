# 使用 docker build -t gin-user-manage . 构建镜像
# 使用 docker run -d -p 8080:8080 gin-user-manage 运行容器

# 第一阶段：构建应用
FROM golang:1.23-alpine AS builder

# 设置Go环境变量
ENV GO111MODULE=on \
    CGO_ENABLED=0 \
    GOOS=linux \
    GOARCH=amd64 \
    GOPROXY=https://goproxy.cn,direct

# 设置工作目录
WORKDIR /app

# 复制 go.mod 和 go.sum 文件
COPY go.mod ./
COPY go.sum ./

# 下载依赖
RUN go mod download

# 复制源代码
COPY . .

RUN go build -o main .


# 第二阶段：创建最小运行镜像
FROM alpine:latest


WORKDIR /root/

# 从构建阶段复制编译好的应用
COPY --from=builder /app/main .
COPY --from=builder /app/user.json .

# 暴露 8080 端口
EXPOSE 8080

# 运行应用
CMD ["./main"]