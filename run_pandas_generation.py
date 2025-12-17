#!/usr/bin/env python3
"""
CodeAgent pandas 任务生成脚本（实际运行）

这个脚本真实地运行 CodeAgent，为 pandas 仓库生成训练任务。
环境变量已提前设置，沙盒会自动启动和关闭。
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CodeAgent pandas 任务生成')
    parser.add_argument('--taskdb-root', default='pandas_taskdb', help='任务保存目录')
    parser.add_argument('--repo-url', default='https://github.com/pandas-dev/pandas', help='仓库 URL')
    parser.add_argument('--difficulty', type=int, default=2, help='任务难度 (1-3)')
    parser.add_argument('--agent-type', default='code_agent', help='Agent 类型')
    parser.add_argument('--num-tasks', type=int, default=1, help='生成任务数量')
    parser.add_argument('--log-level', default='INFO', help='日志级别')
    parser.add_argument('--max-tokens', type=int, default=2000, help='LLM 最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.7, help='LLM 采样温度 (0.0-1.0)')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    print("\n" + "=" * 80)
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "CodeAgent pandas 任务生成脚本" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    
    print("\n" + "=" * 80)
    print("🚀 开始 pandas 任务生成")
    print("=" * 80)
    
    # 导入必需模块
    try:
        from agent_gem.llm import LLMClient
        from agent_gem.agents.code_agent import CodeAgent
        from agent_gem.generator import GenerationRequest
        logger.info("✅ 模块导入成功")
    except ImportError as e:
        logger.error(f"❌ 模块导入失败: {e}")
        return 1
    
    # 初始化 LLM
    print("\n📌 初始化 LLM 客户端...")
    try:
        llm = LLMClient.from_env()
        logger.info("✅ LLM 客户端初始化成功")
    except Exception as e:
        logger.error(f"❌ LLM 客户端初始化失败: {e}")
        return 1
    
    # 创建 CodeAgent
    print("📌 创建 CodeAgent...")
    try:
        agent = CodeAgent(
            llm, 
            taskdb_root=args.taskdb_root,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        logger.info("✅ CodeAgent 创建成功")
        logger.info(f"   - 类型: {agent.agent_type}")
        logger.info(f"   - 描述: {agent.description}")
        logger.info(f"   - LLM 最大生成长度: {agent.max_tokens}")
        logger.info(f"   - 采样温度: {agent.temperature}")
    except Exception as e:
        logger.error(f"❌ CodeAgent 创建失败: {e}")
        return 1
    
    # 创建生成请求
    print("📌 创建生成请求...")
    try:
        request = GenerationRequest(
            agent_type=args.agent_type,
            topic=args.repo_url,
            difficulty=args.difficulty,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        logger.info("✅ 生成请求创建成功")
        logger.info(f"   - 仓库: {args.repo_url}")
        logger.info(f"   - 难度: {args.difficulty}")
        logger.info(f"   - 最大生成长度: {args.max_tokens}")
        logger.info(f"   - 采样温度: {args.temperature}")
    except Exception as e:
        logger.error(f"❌ 生成请求创建失败: {e}")
        return 1
    
    # 执行任务生成
    print("\n📌 执行任务生成流程...")
    print("   这将执行以下步骤:")
    print("   1. 启动 SandboxFusion 容器")
    print("   2. 克隆 pandas GitHub 仓库")
    print("   3. 分析仓库（语言、依赖、测试框架）")
    print("   4. 提取源代码文件")
    print("   5. 使用 LLM 生成合成 bug")
    print("   6. 生成 GitHub issue/PR 描述")
    print("   7. 生成测试用例")
    print("   8. 在沙盒验证测试")
    print("   9. 生成环境配置")
    print("   10. 保存结构化任务数据")
    
    logger.info("\n开始生成任务...")
    try:
        package = agent.generate(request)
        
        if package is None:
            logger.error("❌ 任务生成返回 None")
            print("\n" + "=" * 80)
            print("❌ 任务生成失败")
            print("=" * 80)
            print("\n可能的原因:")
            print("  - LLM API 错误或 API 额度不足")
            print("  - SandboxFusion 服务未运行或无法连接")
            print("  - 网络连接问题（克隆 pandas 需要较大带宽）")
            print("  - 磁盘空间不足（pandas 仓库较大）")
            return 1
        
        # 显示生成结果
        print("\n" + "=" * 80)
        print("✅ 任务生成成功!")
        print("=" * 80)
        
        print(f"\n📋 任务详情:")
        print(f"   - 任务 ID: {package.task.task_id}")
        print(f"   - 任务标题: {package.task.task_title}")
        print(f"   - 任务内容: {package.task.task_content[:100]}...")
        print(f"   - 难度级别: {package.task.difficulty_level}")
        
        print(f"\n💾 任务保存位置:")
        task_dir = Path(args.taskdb_root) / f"task_{package.task.task_id}"
        print(f"   {task_dir}/")
        print(f"   ├── task.json          # 任务定义")
        print(f"   ├── solution.txt       # 修复代码和测试")
        print(f"   ├── context.json       # 元数据")
        print(f"   └── task_info.txt      # 摘要信息")
        
        print(f"\n🎯 下一步:")
        print(f"   1. 查看生成的任务: cat {task_dir}/task_info.txt")
        print(f"   2. 检查 JSON 数据: cat {task_dir}/task.json | jq")
        print(f"   3. 生成更多任务: NUM_TASKS=5 bash ./run_code_agent.sh")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 任务生成失败: {e}", exc_info=True)
        print("\n" + "=" * 80)
        print("❌ 任务生成出错")
        print("=" * 80)
        print(f"\n错误信息: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
