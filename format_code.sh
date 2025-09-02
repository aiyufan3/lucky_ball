#!/bin/bash

# 代码格式化脚本
# 使用 black 和 isort 来格式化 Python 代码

echo "🔧 开始格式化 Python 代码..."

# 检查是否安装了必要的工具
if ! python3 -c "import black" 2>/dev/null; then
    echo "❌ 未安装 black，正在安装..."
    python3 -m pip install black
fi

if ! python3 -c "import isort" 2>/dev/null; then
    echo "❌ 未安装 isort，正在安装..."
    python3 -m pip install isort
fi

echo "📁 格式化 scripts/ 目录..."
python3 -m black --line-length 88 scripts/
python3 -m isort scripts/

echo "📁 格式化 test/ 目录..."
python3 -m black --line-length 88 test/
python3 -m isort test/

echo "✅ 代码格式化完成！"

# 检查格式是否正确
echo "🔍 检查代码格式..."
python3 -m black --check --line-length 88 scripts/ test/

if [ $? -eq 0 ]; then
    echo "🎉 所有代码格式正确！"
else
    echo "⚠️  发现格式问题，请运行格式化脚本修复"
fi
