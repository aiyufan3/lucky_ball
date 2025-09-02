# 开发指南

## 代码格式化

本项目使用以下工具来保持代码质量：

### 格式化工具
- **Black**: Python代码格式化器，行长度88字符
- **isort**: 自动整理import语句
- **flake8**: 代码风格检查

### 快速格式化
```bash
# 使用提供的脚本
./format_code.sh

# 或手动运行
python3 -m black --line-length 88 scripts/ test/
python3 -m isort scripts/ test/
```

### 配置
- `pyproject.toml`: 包含black和isort的配置
- 行长度: 88字符
- Python版本: 3.9+

## Git配置

### 项目级用户配置
本项目使用专用的Git用户配置：

```bash
git config user.name "luckyluckybot"
git config user.email "luckyluckybot@example.com"
```

这些配置仅适用于本项目，不会影响全局Git设置。

### GitHub Actions
工作流自动使用项目级Git配置进行提交，确保所有自动提交都使用统一的用户信息。

## 开发流程

1. **代码编写**: 遵循PEP 8规范
2. **格式化**: 运行 `./format_code.sh`
3. **测试**: 运行测试脚本验证功能
4. **提交**: 使用项目级Git配置

## 依赖管理

### 安装依赖
```bash
python3 -m pip install -r requirements.txt
```

### 开发依赖
```bash
python3 -m pip install black isort flake8 pytest
```

## 代码质量检查

### 格式检查
```bash
python3 -m black --check --line-length 88 scripts/ test/
```

### 导入排序检查
```bash
python3 -m isort --check-only scripts/ test/
```

### 风格检查
```bash
python3 -m flake8 scripts/ test/
```

## 注意事项

- 所有Python文件都应该通过black格式化
- import语句应该通过isort整理
- 提交前请运行格式化脚本
- 保持代码风格一致性
