
# 使用官方 Python 3.11 镜像
FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 包管理工具
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 安装 pip-tools
RUN pip install --no-cache-dir pip-tools

# 设置工作目录
WORKDIR /app

# 复制代码文件
COPY . .

# 1) 使用 pip-compile 基于 requirements_no_version.txt 解析锁定兼容版本
RUN pip-compile requirements_no_version.txt --resolver=backtracking --output-file=requirements.txt

# 2) 检查是否有 torch 相关包，分别处理
RUN python split_torch_packages.py

# 3) 先安装非 torch 包
RUN if [ -f "requirements_other.txt" ]; then \
        echo "Installing non-PyTorch packages..."; \
        pip install --no-cache-dir --prefer-binary -r requirements_other.txt; \
    fi

# 4) 再安装 torch 相关包（使用 CPU 索引）
RUN if [ -f "requirements_torch.txt" ]; then \
        echo "Installing PyTorch packages with CPU index..."; \
        pip install --no-cache-dir --prefer-binary --extra-index-url https://download.pytorch.org/whl/cpu -r requirements_torch.txt; \
    fi

CMD ["python", "main.py"]
