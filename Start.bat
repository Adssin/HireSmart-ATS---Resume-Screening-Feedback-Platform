@echo off

REM ====== START ML API ======
cd /d "D:\1PROJECTS\SmartHireATS\SmartHireATS\backend(ML)\ml_api.py"
start cmd /k "uvicorn ml_api:app --reload"

REM ====== START FRONTEND ======
cd /d "D:\1PROJECTS\SmartHireATS\SmartHireATS\frontend"
start cmd /k "npm run dev"