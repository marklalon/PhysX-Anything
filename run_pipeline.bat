@echo off
:: PhysX-Anything 推理管线一键脚本
:: 用法:
::   run_pipeline.bat [图片完整路径]                       # 正常模式（VRAM >= 16 GB）
::   run_pipeline.bat [图片完整路径] --load_in_8bit True   # 8-bit 量化（VRAM < 16 GB）
:: 示例:
::   run_pipeline.bat "D:\AI\ComfyUI\output\ComfyUI_00044_.png"
::   run_pipeline.bat "D:\AI\ComfyUI\output\ComfyUI_00044_.png" --load_in_8bit True

setlocal enabledelayedexpansion

set REPO_DIR=D:\AI\PhysX-Anything
set INPUT_IMAGE=%~1

:: 激活 venv
call "%REPO_DIR%\venv\Scripts\activate.bat"
cd /d "%REPO_DIR%"

:: 复制图片到 demo/
if not "%INPUT_IMAGE%"=="" (
    if not exist "%REPO_DIR%\demo\" mkdir "%REPO_DIR%\demo"
    copy /y "%INPUT_IMAGE%" "%REPO_DIR%\demo\" >nul
    echo [*] 已复制 %~nx1 到 .\demo\
)

:: 提取图片文件名
set IMAGE_NAME=%~nx1

:: 拼接额外参数（%2 及之后）
set EXTRA_ARGS=
shift
:collect_args
if "%~1"=="" goto run
set EXTRA_ARGS=%EXTRA_ARGS% %1
shift
goto collect_args

:run
echo.
echo [1/4] VLM 分析中（Qwen2.5-VL-7B）...
python 1_vlm_demo.py --demo_path ./demo --ckpt ./pretrain/vlm --save_part_ply True --remove_bg False --image "%IMAGE_NAME%" %EXTRA_ARGS%
if errorlevel 1 goto error

echo.
echo [2/4] TRELLIS 3D 解码中...
python 2_decoder.py
if errorlevel 1 goto error

echo.
echo [3/4] 网格分割中...
python 3_split.py --index 0 --range 2000
if errorlevel 1 goto error

echo.
echo [4/4] 生成仿真文件（URDF/MJCF）...
python 4_simready_gen.py --basepath ./test_demo --voxel_define 32 --process 0 --fixed_base 0 --deformable 0
if errorlevel 1 goto error

echo.
echo ==========================================
echo 完成！结果目录: %REPO_DIR%\test_demo\
echo   - sample.glb    ^→ 3D 网格（可用 Blender 打开）
echo   - basic.urdf    ^→ 仿真文件
echo   - basic.xml     ^→ MuJoCo 格式
echo ==========================================
goto end

:error
echo.
echo [错误] 管线执行失败，请检查上方错误信息。
exit /b 1

:end
endlocal
