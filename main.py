import json
import subprocess
import sys
import traceback
import os
from pathlib import Path

def lambda_handler(event, context):
    """
    AWS Lambda handler function that executes experiment.py as a subprocess
    """
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        
        # 获取experiment.py的路径
        experiment_path = Path(__file__).parent / "experiment.py"
        
        # 检查文件是否存在
        if not experiment_path.exists():
            raise FileNotFoundError(f"experiment.py not found at {experiment_path}")
        
        # 使用subprocess执行experiment.py
        result = subprocess.run(
            [sys.executable, str(experiment_path)],
            capture_output=True,
            text=True,
            timeout=840,  # 14分钟超时，留1分钟给Lambda清理
            env=env,
            cwd=str(experiment_path.parent)  # 设置工作目录
        )
        
        # 强信号 + 导入失败类型（大小写不敏感）
        has_error = result.returncode != 0
        if not has_error:
            combined_output = (result.stdout or "") + (result.stderr or "")
            combined_output_lower = combined_output.lower()
            strong_error_signals = [
                "traceback",
                "exception:",
                "experiment failed",
                "importerror",
                "modulenotfounderror",
                "no module named",
                "cannot import name",
            ]
            if any(sig in combined_output_lower for sig in strong_error_signals):
                has_error = True
        
        response = {
            'experimentStatusCode': 500 if has_error else 200,
            'body': {
                'message': 'Experiment failed' if has_error else 'Experiment completed successfully',
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': context.get_remaining_time_in_millis() if context else None
            }
        }
        
        # 输出执行结果
        if result.stdout:
            print("=== Experiment Output ===")
            print(result.stdout)
        
        if result.stderr:
            print("=== Experiment Errors ===")
            print(result.stderr)
        
        return response
        
    except subprocess.TimeoutExpired:
        error_response = {
            'statusCode': 500,  # 改为500，不再使用408
            'body': {
                'message': 'Experiment execution timeout',
                'error': 'Process exceeded timeout limit'
            }
        }
        
        print("Experiment execution timeout")
        return error_response
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        
        error_response = {
            'statusCode': 500,
            'body': {
                'message': 'Lambda handler failed',
                'error': str(e),
                'traceback': error_traceback
            }
        }
        
        print(f"Lambda handler failed with error: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return error_response

if __name__ == "__main__": 
    # 直接执行时的逻辑 
    result = lambda_handler({}, None) 
    print(json.dumps(result, indent=2))
        
        